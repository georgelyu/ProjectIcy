from __future__ import annotations

import io
import unittest
from unittest import mock

from iceflow2d.simulator import _format_duration, _TerminalProgress


class ProgressTests(unittest.TestCase):
    def test_duration_format(self):
        self.assertEqual(_format_duration(None), "--:--")
        self.assertEqual(_format_duration(65.0), "01:05")
        self.assertEqual(_format_duration(3661.0), "01:01:01")

    def test_progress_reaches_completion(self):
        stream = io.StringIO()
        with mock.patch(
            "iceflow2d.simulator.time.monotonic", side_effect=(100.0, 102.0, 105.0)
        ):
            with _TerminalProgress(2, enabled=True, stream=stream, width=4) as progress:
                progress.advance()
                progress.advance()

        output = stream.getvalue()
        self.assertIn("[----]", output)
        self.assertIn("1/2 frames", output)
        self.assertIn("[####]", output)
        self.assertIn("100.0%", output)
        self.assertIn("2/2 frames", output)
        self.assertTrue(output.endswith("\n"))

    def test_disabled_and_zero_progress_are_silent(self):
        for total, enabled in ((2, False), (0, True)):
            with self.subTest(total=total, enabled=enabled):
                stream = io.StringIO()
                with _TerminalProgress(
                    total, enabled=enabled, stream=stream
                ) as progress:
                    progress.advance()
                self.assertEqual(stream.getvalue(), "")

    def test_exception_adds_newline_without_suppressing_error(self):
        stream = io.StringIO()
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            with _TerminalProgress(2, enabled=True, stream=stream) as progress:
                progress.advance()
                raise RuntimeError("stopped")

        self.assertIn("1/2 frames", stream.getvalue())
        self.assertTrue(stream.getvalue().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
