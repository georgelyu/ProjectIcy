// Optional execution adapter for Taichi 1.7.4 no-argument field kernels.
// Observe public CUPTI driver callbacks; recreate the recorded order with
// public CUDA graph APIs. No solver kernels or legacy code are introduced.
// https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html
#include <cuda.h>
#include <cupti.h>
#include <map>
#include <set>
#include <string>
#include <mutex>
#include <cstdio>
#include <array>
#include <vector>
#include <cstring>
#include <cstdint>

static CUpti_SubscriberHandle subscriber;
static std::mutex guard;
static std::map<std::string, unsigned> calls;
static std::map<CUfunction, std::string> functions;
static std::set<CUstream> streams;
static CUcontext context = nullptr;
static CUgraph graph = nullptr;
static CUgraphExec executable = nullptr;
static CUstream stream = nullptr;
struct Event {
  int kind; // kernel, allocation, free
  CUDA_KERNEL_NODE_PARAMS launch{};
  std::array<uint64_t, 4> arguments{};
  CUdeviceptr address = 0;
  size_t bytes = 0;
};
static std::vector<Event> events;
static std::set<CUfunction> validated;
static bool recording = false;
static std::string recording_error;

static int checked(CUresult status) {
  if (status != CUDA_SUCCESS) {
    const char *name = nullptr, *description = nullptr;
    cuGetErrorName(status, &name); cuGetErrorString(status, &description);
    std::fprintf(stderr, "CUDA %s: %s\n", name, description);
  }
  return static_cast<int>(status);
}

static void CUPTIAPI observe(void *, CUpti_CallbackDomain domain,
                             CUpti_CallbackId id, const void *data) {
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
  const auto *info = static_cast<const CUpti_CallbackData *>(data);
  if (recording) {
    std::lock_guard<std::mutex> lock(guard);
    if (id == CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync && info->callbackSite == CUPTI_API_EXIT) {
      const auto *p = static_cast<const cuMemAllocAsync_params *>(info->functionParams);
      Event event{}; event.kind = 1; event.address = *p->dptr; event.bytes = p->bytesize;
      if (p->hStream != stream || *static_cast<const CUresult *>(info->functionReturnValue) != CUDA_SUCCESS)
        recording_error = "allocation stream/status mismatch";
      events.push_back(event);
    }
    if (info->callbackSite != CUPTI_API_ENTER) return;
    if (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
      const auto *p = static_cast<const cuLaunchKernel_params *>(info->functionParams);
      if (!validated.count(p->f) || p->extra || !p->kernelParams || p->hStream != stream) {
        recording_error = "kernel ABI or stream mismatch"; return;
      }
      Event event{}; event.kind = 0;
      event.launch.func = p->f;
      event.launch.gridDimX = p->gridDimX; event.launch.gridDimY = p->gridDimY; event.launch.gridDimZ = p->gridDimZ;
      event.launch.blockDimX = p->blockDimX; event.launch.blockDimY = p->blockDimY; event.launch.blockDimZ = p->blockDimZ;
      event.launch.sharedMemBytes = p->sharedMemBytes;
      std::memcpy(event.arguments.data(), p->kernelParams[0], 32);
      events.push_back(event);
    } else if (id == CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync) {
      const auto *p = static_cast<const cuMemFreeAsync_params *>(info->functionParams);
      Event event{}; event.kind = 2; event.address = p->dptr;
      if (p->hStream != stream) recording_error = "free stream mismatch";
      events.push_back(event);
    } else {
      std::string name(info->functionName);
      if (name != "cuCtxGetCurrent" && name != "cuCtxSetCurrent" && name != "cuModuleGetFunction" &&
          name != "cuStreamSynchronize" && name != "cuMemAllocAsync")
        recording_error = "unsupported driver call: "+name;
    }
    return;
  }
  if (info->callbackSite != CUPTI_API_ENTER) return;
  std::lock_guard<std::mutex> lock(guard);
  ++calls[info->functionName];
  if (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
      id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
    const auto *params = static_cast<const cuLaunchKernel_params *>(info->functionParams);
    functions[params->f] = info->symbolName ? info->symbolName : "unknown";
    streams.insert(params->hStream);
    context = info->context;
  }
}

extern "C" int begin_probe() {
  if (!recording) { calls.clear(); functions.clear(); streams.clear(); validated.clear(); }
  auto result = cuptiSubscribe(&subscriber, observe, nullptr);
  if (result != CUPTI_SUCCESS) return static_cast<int>(result);
  return static_cast<int>(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
}

extern "C" int end_probe() {
  auto result = cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  cuptiUnsubscribe(subscriber);
  for (const auto &[name, count] : calls) std::printf("API %u %s\n", count, name.c_str());
  std::printf("FUNCTIONS %zu STREAMS %zu\n", functions.size(), streams.size());
  if (streams.size() == 1) stream = *streams.begin();
  if (context && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
    std::map<std::string, unsigned> signatures;
    for (const auto &[function, name] : functions) {
      std::string signature;
      for (size_t index = 0; index < 64; ++index) {
        size_t offset = 0, size = 0;
        auto status = cuFuncGetParamInfo(function, index, &offset, &size);
        if (status == CUDA_ERROR_INVALID_VALUE) break;
        if (status != CUDA_SUCCESS) { signature += "ERROR"; break; }
        signature += std::to_string(offset)+":"+std::to_string(size)+",";
      }
      ++signatures[signature];
      if (signature == "0:32,") validated.insert(function);
    }
    for (const auto &[signature, count] : signatures) std::printf("SIGNATURE %u %s\n", count, signature.c_str());
    CUcontext previous;
    cuCtxPopCurrent(&previous);
  }
  std::fflush(stdout);
  return static_cast<int>(result);
}

extern "C" int record_begin() {
  events.clear(); recording_error.clear(); recording = true;
  return begin_probe();
}

extern "C" int record_end(int elide_unused_return_buffers) {
  cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  cuptiUnsubscribe(subscriber); recording = false;
  if (!recording_error.empty()) {
    std::fprintf(stderr, "%s\n", recording_error.c_str()); return -1;
  }
  // Taichi 1.7.4 allocates an unused eight-byte return buffer even for a
  // no-argument, void field kernel. Elide ONLY when every allocation has that
  // size, no recorded kernel argument refers to it, and the strict API allow
  // list above proves there was no host/device copy publishing its address.
  std::set<CUdeviceptr> temporary_addresses;
  bool unused = elide_unused_return_buffers != 0;
  for (const auto &event : events) if (event.kind == 1) {
    temporary_addresses.insert(event.address);
    if (event.bytes != 8) unused = false;
  }
  for (const auto &event : events) if (event.kind == 0)
    for (auto word : event.arguments) if (temporary_addresses.count(word)) unused = false;
  auto status = cuCtxPushCurrent(context);
  if (status != CUDA_SUCCESS) return checked(status);
  CUdevice device; cuCtxGetDevice(&device);
  status = cuGraphCreate(&graph, 0);
  CUgraphNode previous = nullptr;
  std::map<CUdeviceptr, CUdeviceptr> live;
  for (auto &event : events) {
    if (status != CUDA_SUCCESS) break;
    if (unused && event.kind != 0) continue;
    CUgraphNode node = nullptr;
    const auto *dependencies = previous ? &previous : nullptr;
    const size_t count = previous ? 1 : 0;
    if (event.kind == 1) {
      CUDA_MEM_ALLOC_NODE_PARAMS params{};
      params.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
      params.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      params.poolProps.location.id = device;
      params.bytesize = event.bytes;
      status = cuGraphAddMemAllocNode(&node, graph, dependencies, count, &params);
      if (live.count(event.address)) { status = CUDA_ERROR_INVALID_VALUE; break; }
      live[event.address] = params.dptr;
    } else if (event.kind == 2) {
      auto found = live.find(event.address);
      if (found == live.end()) { status = CUDA_ERROR_INVALID_VALUE; break; }
      status = cuGraphAddMemFreeNode(&node, graph, dependencies, count, found->second);
      live.erase(found);
    } else {
      auto copied = event.arguments;
      for (auto &word : copied) {
        auto found = live.find(word);
        if (found != live.end()) word = found->second;
      }
      void *parameters[] = {copied.data()};
      event.launch.kernelParams = parameters;
      status = cuGraphAddKernelNode(&node, graph, dependencies, count, &event.launch);
      event.launch.kernelParams = nullptr;
    }
    previous = node;
  }
  if (!live.empty()) status = CUDA_ERROR_INVALID_VALUE;
  if (status == CUDA_SUCCESS) status = cuGraphInstantiateWithFlags(&executable, graph, 0);
  std::printf("RECORDED GRAPH EVENTS %zu UNUSED RETURN BUFFERS ELIDED %d STATUS %d\n", events.size(), unused, static_cast<int>(status));
  CUcontext old; cuCtxPopCurrent(&old);
  return checked(status);
}

extern "C" int capture_begin() {
  if (!context || !stream) return -1;
  auto status = cuCtxPushCurrent(context);
  if (status != CUDA_SUCCESS) return checked(status);
  status = cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
  CUcontext old;
  cuCtxPopCurrent(&old);
  return checked(status);
}

extern "C" int capture_end() {
  auto status = cuCtxPushCurrent(context);
  if (status != CUDA_SUCCESS) return checked(status);
  status = cuStreamEndCapture(stream, &graph);
  if (status == CUDA_SUCCESS) status = cuGraphInstantiateWithFlags(&executable, graph, 0);
  if (status == CUDA_SUCCESS) {
    size_t nodes = 0;
    cuGraphGetNodes(graph, nullptr, &nodes);
    std::printf("GRAPH NODES %zu\n", nodes);
    std::fflush(stdout);
  }
  CUcontext old;
  cuCtxPopCurrent(&old);
  return checked(status);
}

extern "C" int launch_graph(int count) {
  auto status = cuCtxPushCurrent(context);
  if (status != CUDA_SUCCESS) return checked(status);
  for (int i = 0; i < count && status == CUDA_SUCCESS; ++i)
    status = cuGraphLaunch(executable, stream);
  if (status == CUDA_SUCCESS) status = cuStreamSynchronize(stream);
  CUcontext old;
  cuCtxPopCurrent(&old);
  return checked(status);
}

extern "C" int destroy_graph() {
  auto status = cuCtxPushCurrent(context);
  if (status != CUDA_SUCCESS) return checked(status);
  if (executable) status = cuGraphExecDestroy(executable);
  if (graph) {
    auto released = cuGraphDestroy(graph);
    if (status == CUDA_SUCCESS) status = released;
  }
  executable = nullptr; graph = nullptr;
  CUcontext old; cuCtxPopCurrent(&old);
  return checked(status);
}
