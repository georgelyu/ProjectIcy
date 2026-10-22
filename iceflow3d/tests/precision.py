"""Host-side error budgets for storage rounded to IEEE binary32."""

import numpy as np


def exchange_rounding_budget(sim, energy):
    """Half an energy ULP per cell eligible to exchange heat with ice.

    The contact mask is independently constructed on the host. This budget
    covers rounding stored energy; it does not allow missing heat fluxes or
    a change to the temperature/latent-heat constitutive equations.
    """
    contact = (sim.coverage_body_id.to_numpy() >= 0) & (sim.coverage.to_numpy() > 1e-8)
    near = contact.copy()
    for axis in range(3):
        near |= np.roll(contact, 1, axis) | np.roll(contact, -1, axis)
    mask = (
        near
        & (sim.solid_body_id.to_numpy() == -1)
        & (sim.water_volume.to_numpy() > 1e-30)
    )
    return 0.5 * np.spacing(energy[mask]).astype(float).sum()
