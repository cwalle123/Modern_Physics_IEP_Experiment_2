"""Imports"""

# External imports
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#################################################################################################################################################################
"""Plot styling"""

# Increase font sizes across every figure (labels, ticks, legends) so plots
# stay legible once shrunk down to fit the report's column width.
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 16,
})

#################################################################################################################################################################
"""Constants"""

Hg_N_lines_per_mm = 1800    # lines/mm  (grating used for this setup)
Na_N_lines_per_mm = 1800    # lines/mm  (grating used for this setup)

reading_uncertainty_deg = 1                                                 # deg  (estimated reading uncertainty on the angle scale, per line)
angle_scale_at_i_is_0_Hg = 53                                               # deg  (when i = 0, the angle scale reads 53 +-1 deg. This is the 0 point)
angle_scale_at_i_is_0_Na = 53                                               # deg  (when i = 0, the angle scale reads 53 +-1 deg. This is the 0 point)
angle_scale_at_0th_order_Hg = 63                                            # deg  (At the 0th order, the angle scale reads 63 +-1 deg)
angle_scale_at_0th_order_Na = 62                                            # deg  (At the 0th order, the angle scale reads 62 +-1 deg)
alpha_deg_Hg = abs(angle_scale_at_0th_order_Hg - angle_scale_at_i_is_0_Hg)  # deg
alpha_deg_Na = abs(angle_scale_at_0th_order_Na - angle_scale_at_i_is_0_Na)  # deg
alpha_uncertainty_deg = 2 * reading_uncertainty_deg                         # deg  (estimated reading uncertainty on the angle scale)

camera_pixel_size_m = 3.45 * 10**-6                                         # m / pixel
Na_pixels_between_yellow_1_and_yellow_2 = 116                               # pixels (measured from the camera image)
Hg_pixels_between_yellow_1_and_yellow_2 = 101                               # pixels (measured from the camera image)
pixel_reading_uncertainty_px = 5                                            # px  (estimated uncertainty on each line's pixel position on the camera)

f2_focal_length_m = 0.100  # m  (focal length of imaging lens f2, used to convert the camera pixel separation of the doublet into an angular separation Delta_u)

grating_width_mm = 50  # mm  (total grating width)

# Literature values (nm), for the agreement checks and the plots.
HG_LITERATURE_NM = {
    'violet': 404.66,
    'blue': 435.83,
    'green': 546.07,
    'yellow1': 576.96,
    'yellow2': 579.07,
}
NA_LITERATURE_NM = {
    'yellow1': 589.00,
    'yellow2': 589.59,
}

# Lamp-dependent working constants. These are (re)populated by configure_lamp()
lamp = None
N_lines_per_mm = None
angle_scale_at_0th_order = None
alpha_deg = None
N_per_m = None
LITERATURE_NM = None
doublet_pixels = None
N_tot = None

def configure_lamp(lamp_name):
    """
    Set every lamp-dependent module-level constant for lamp_name ('Hg' or
    'Na'). Must be called before any of the calculation functions below,
    and again to switch lamps -- this is what main() does in its loop.
    """
    global lamp, N_lines_per_mm, angle_scale_at_0th_order, alpha_deg
    global N_per_m, LITERATURE_NM, doublet_pixels, N_tot

    lamp = lamp_name
    N_lines_per_mm = Hg_N_lines_per_mm if lamp == 'Hg' else Na_N_lines_per_mm
    angle_scale_at_0th_order = angle_scale_at_0th_order_Hg if lamp == 'Hg' else angle_scale_at_0th_order_Na
    alpha_deg = alpha_deg_Hg if lamp == 'Hg' else alpha_deg_Na
    N_per_m = N_lines_per_mm * 1e3  # lines/m
    LITERATURE_NM = HG_LITERATURE_NM if lamp == 'Hg' else NA_LITERATURE_NM
    doublet_pixels = Hg_pixels_between_yellow_1_and_yellow_2 if lamp == 'Hg' else Na_pixels_between_yellow_1_and_yellow_2
    N_tot = N_lines_per_mm * grating_width_mm  # total illuminated grating lines

#################################################################################################################################################################
"""Functions"""

def _style_axes(ax, include_x_zero=False, include_y_zero=False):
    """
    Apply consistent, rubric-compliant styling to a figure axis:

    - No in-figure title (the caption belongs in the report text, not the
      plot itself).
    - Tick spacing restricted to legible steps of 1, 2 or 5 (via
      MaxNLocator), instead of matplotlib's arbitrary default spacing.
    - A light grid, since it makes it easier to read values off the axes
      precisely -- helpful in both colour and black-and-white printouts.
    - Optionally forces 0 onto an axis where that is a meaningful physical
      reference point.

    ax : a matplotlib Axes object
    """
    ax.set_title('')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    if include_x_zero:
        left, right = ax.get_xlim()
        ax.set_xlim(left=min(0, left), right=right)
    if include_y_zero:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom=min(0, bottom), top=top)
    ax.grid(True, linewidth=0.4, alpha=0.5)

def load_data(filepath):
    """
    Read the raw angle readings from a CSV file.

    Expected columns: colour, order, reading (with a header row to skip).
    skipinitialspace=True and the extra .strip() handle "colour, order,
    reading"-style headers/rows where a space follows each comma.

    Returns a list of tuples: (colour, order, reading_deg)
    """
    runs = []
    with open(filepath, newline='') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)  # skip header row
        for row in reader:
            if not row:  # skip blank lines (e.g. trailing newline at end of file)
                continue
            colour, order, reading_deg = row
            # Convert order and reading to their proper types before storing
            runs.append((colour.strip(), int(order), float(reading_deg)))
    return runs

def group_by_colour(runs):
    """
    Group angle readings by colour (i.e. by which spectral line was measured).
    Needed because the weighted average (compute_group_results) is taken
    per spectral line, over every order that line was measured at -- it
    can't be taken across colours, since those are genuinely different
    wavelengths.

    runs : list of (colour, order, reading_deg) tuples

    Returns a dict mapping colour -> list of (order, reading_deg) tuples for
    that colour.
    """
    groups = {}
    for colour, order, reading_deg in runs:
        # setdefault creates an empty list the first time this colour is seen
        groups.setdefault(colour, []).append((order, reading_deg))
    return groups


def get_phi(reading_deg):
    """
    phi is the angle between a given line's reading and the 0th-order
    ('mirror') reading -- i.e. the raw angle-scale value recorded when the
    grating was rotated so the 0th order landed on the chosen point

    This is NOT alpha_deg: alpha is a different quantity (how far the
    grating sits from the separate i=0 calibration reading), not phi's
    zero point on the scale.

    reading_deg : angle scale reading for this spectral line (deg)

    Returns phi in degrees.
    """
    return reading_deg - angle_scale_at_0th_order

def get_wavelength(order, reading_deg):
    """
    Compute lambda from 2*cos(alpha)*sin(phi) = -m*N*lambda

    order       : diffraction order m (nonzero integer)
    reading_deg : angle scale reading for this spectral line (deg)

    Returns lambda in nm.
    """
    phi_rad = math.radians(get_phi(reading_deg))
    alpha_rad = math.radians(alpha_deg)
    # Formula gives lambda in metres directly, since N_per_m is lines/m
    lambda_m = -2 * math.cos(alpha_rad) * math.sin(phi_rad) / (order * N_per_m)
    # Convert m -> nm
    return lambda_m * 1e9

def get_wavelength_uncertainty(order, reading_deg):
    """
    Propagate uncertainty in alpha_deg and phi (= reading_deg - alpha_deg)
    into an uncertainty on lambda, using exact partial-derivative
    (first-order) error propagation:

        u_lambda^2 = (dlambda/dalpha)^2 * u_alpha^2 + (dlambda/dphi)^2 * u_phi^2

    From lambda = -2*cos(alpha)*sin(phi) / (m*N):

        dlambda/dalpha =  2*sin(alpha)*sin(phi) / (m*N)
        dlambda/dphi   = -2*cos(alpha)*cos(phi) / (m*N)

    phi's own uncertainty combines the line reading and the zero-point
    reading (angle_scale_at_0th_order) in quadrature, since both are
    independent raw angle-scale readings with the same instrument
    uncertainty:

        u_phi = sqrt(reading_uncertainty_deg^2 + reading_uncertainty_deg^2)

    This treats the input uncertainties as independent (adding their
    contributions in quadrature), unlike a simple sum, which implicitly
    assumes worst-case correlated errors and overestimates u_lambda.

    order       : diffraction order m (nonzero integer)
    reading_deg : angle scale reading for this spectral line (deg)

    Returns the uncertainty in lambda, in nm.
    """
    alpha_rad = math.radians(alpha_deg)
    phi_rad = math.radians(get_phi(reading_deg))
    u_alpha_rad = math.radians(alpha_uncertainty_deg)
    u_phi_rad = math.radians(math.hypot(reading_uncertainty_deg, reading_uncertainty_deg))

    # Partial derivatives of lambda (in m) with respect to each input
    dlambda_dalpha = 2 * math.sin(alpha_rad) * math.sin(phi_rad) / (order * N_per_m)
    dlambda_dphi = -2 * math.cos(alpha_rad) * math.cos(phi_rad) / (order * N_per_m)

    # Combine contributions in quadrature (independent-error propagation)
    lambda_m_uncertainty = math.sqrt(
        (dlambda_dalpha * u_alpha_rad) ** 2 +
        (dlambda_dphi * u_phi_rad) ** 2
    )

    # Convert m -> nm
    return lambda_m_uncertainty * 1e9

def get_u(reading_deg):
    """
    u is the angle of reflection, u = alpha - phi.
    This is the angle later needed for the line-splitting calculation
    (Delta_lambda = cos(u)/(m*N) * Delta_u).

    Built directly from get_phi (reading_deg - zero_point_reading_deg),
    so it automatically stays consistent with however phi is defined --
    no separate algebraic shortcut to keep in sync.

    reading_deg : angle scale reading for this spectral line (deg)

    Returns u in degrees.
    """
    return alpha_deg - get_phi(reading_deg)

def get_u_uncertainty():
    """
    Propagate uncertainty in alpha_deg, reading_deg and the zero-point
    reading into an uncertainty on u, using exact partial-derivative error
    propagation.

    From u = alpha - phi = alpha - reading + zero_point_reading_deg:

        du/dalpha       =  1
        du/dreading      = -1
        du/dzero_point   =  1

        u_u^2 = u_alpha^2 + u_reading^2 + u_zero_point^2

    zero_point_reading_deg is itself just a raw angle-scale reading, so it
    carries the same reading_uncertainty_deg as any other row. Same
    instrument reading uncertainty applies to every row, so this doesn't
    depend on which line/order you're looking at.

    Returns the uncertainty in u, in degrees.
    """
    return math.sqrt(alpha_uncertainty_deg ** 2 + reading_uncertainty_deg ** 2 + reading_uncertainty_deg ** 2)

def weighted_average(values, uncertainties):
    """
    Weighted average of repeated measurements of the same quantity, per the
    Appendix formulas:

        w_i = 1 / u(lambda_i)^2
        lambda_bar = sum(w_i * lambda_i) / sum(w_i)
        u(lambda_bar) = sqrt(1 / sum(w_i))

    Datapoints with a high uncertainty get a lower weight, and datapoints
    with a low uncertainty get a high weight (are more important).

    values, uncertainties : equal-length lists of lambda_i and u(lambda_i)

    Returns (lambda_bar, u_lambda_bar).
    """
    weights = [1 / u ** 2 for u in uncertainties]
    lambda_bar = sum(w * v for w, v in zip(weights, values)) / sum(weights)
    u_lambda_bar = math.sqrt(1 / sum(weights))
    return lambda_bar, u_lambda_bar

def compute_group_results(groups):
    """
    For each colour group: compute phi, lambda, u and their uncertainties
    for every order measured, print one row per line, and
    compute the weighted average of lambda (per the Appendix formulas)
    across all orders of that colour.

    groups : dict mapping colour -> list of (order, reading_deg) tuples

    Returns two dicts (both keyed by colour):
      group_lambda_bar      : weighted-average lambda per colour (nm)
      group_lambda_bar_unc  : uncertainty on that average per colour (nm)
    """
    group_lambda_bar = {}
    group_lambda_bar_unc = {}
    u_u = get_u_uncertainty()  # same for every row, so only computed once

    print(f"--- {lamp} ---")
    print(f"{'colour':>10} {'order':>6} {'reading':>10} {'phi':>10} {'lambda':>12} "
          f"{'u':>10} {'error in lambda':>16} {'error in u':>12}")

    for colour in sorted(groups):
        lambdas = []
        uncertainties = []
        for order, reading_deg in groups[colour]:
            phi_deg = get_phi(reading_deg)
            lam = get_wavelength(order, reading_deg)
            u_lam = get_wavelength_uncertainty(order, reading_deg)
            u_deg = get_u(reading_deg)

            print(f"{colour:>10} {order:6d} {reading_deg:10.3f} {phi_deg:10.3f} {lam:12.3f} "
                  f"{u_deg:10.3f} {u_lam:16.3f} {u_u:12.3f}")

            lambdas.append(lam)
            uncertainties.append(u_lam)

        lambda_bar, u_lambda_bar = weighted_average(lambdas, uncertainties)
        group_lambda_bar[colour] = lambda_bar
        group_lambda_bar_unc[colour] = u_lambda_bar

    return group_lambda_bar, group_lambda_bar_unc


def print_literature_agreement_table(group_lambda_bar, group_lambda_bar_unc):
    """
    Check each colour's weighted-average lambda against its literature
    value and print a summary table, per the agreement criterion:

        |v| = |a - b| > 2*sqrt(u_a^2 + u_b^2) = 2*u_v  =>  NOT in good agreement

    Literature values are treated as exact (u_literature = 0).

    group_lambda_bar      : dict colour -> weighted-average lambda (nm)
    group_lambda_bar_unc  : dict colour -> uncertainty on that average (nm)
    """
    print(f"{'colour':>10} {'lambda_bar (nm)':>16} {'u':>8} {'literature (nm)':>16} "
          f"{'v':>8} {'2u_v':>8} {'Agreement':>12}")

    for colour in sorted(group_lambda_bar):
        if colour not in LITERATURE_NM:
            continue
        lam_bar = group_lambda_bar[colour]
        u_lam_bar = group_lambda_bar_unc[colour]
        lit = LITERATURE_NM[colour]

        v = abs(lam_bar - lit)
        u_v = 2 * np.sqrt(u_lam_bar ** 2)
        agree = v <= u_v
        verdict = "Agree" if agree else "Disagree"
        print(f"{colour:>10} {lam_bar:16.3f} {u_lam_bar:8.3f} {lit:16.3f} "
              f"{v:8.3f} {u_v:8.3f} {verdict:>12}")


def get_delta_u(pixels):
    """
    Convert the camera-measured pixel separation between two doublet lines
    into an angular separation Delta_u, using the small-angle relation for
    a lens of focal length f2:

        Delta_u ~= (pixels * camera_pixel_size_m) / f2_focal_length_m

    pixels : pixel separation between the two lines on the camera image

    Returns Delta_u in radians.
    """
    return (pixels * camera_pixel_size_m) / f2_focal_length_m

def get_delta_u_uncertainty():
    """
    Propagate the per-line pixel-position uncertainty into an uncertainty
    on Delta_u. The pixel separation is the difference of two independently
    read line positions, each with uncertainty pixel_reading_uncertainty_px,
    so their contributions combine in quadrature:

        u(pixels) = sqrt(pixel_reading_uncertainty_px^2 + pixel_reading_uncertainty_px^2)
        u(Delta_u) = u(pixels) * camera_pixel_size_m / f2_focal_length_m

    Returns the uncertainty in Delta_u, in radians.
    """
    u_pixels = math.hypot(pixel_reading_uncertainty_px, pixel_reading_uncertainty_px)
    return (u_pixels * camera_pixel_size_m) / f2_focal_length_m

def get_delta_lambda(order, reading_deg, delta_u_rad):
    """
    Compute the wavelength splitting of a resolved doublet from eq.
    (deltalabda): Delta_lambda = cos(u)/(m*N) * Delta_u

    order        : diffraction order m of the doublet
    reading_deg  : angle scale reading for this line (used to get u via get_u)
    delta_u_rad  : angular separation of the doublet on the camera (rad)

    Returns Delta_lambda in nm.
    """
    u_rad = math.radians(get_u(reading_deg))
    delta_lambda_m = math.cos(u_rad) / (order * N_per_m) * delta_u_rad
    return delta_lambda_m * 1e9

def get_delta_lambda_uncertainty(order, reading_deg, delta_u_rad, u_delta_u_rad):
    """
    Propagate uncertainty in u and Delta_u into an uncertainty on
    Delta_lambda, using exact partial-derivative error propagation, the
    same way as get_wavelength_uncertainty:

        d(Delta_lambda)/du    = -sin(u)/(m*N) * Delta_u
        d(Delta_lambda)/d(Du) =  cos(u)/(m*N)

    order          : diffraction order m of the doublet
    reading_deg    : angle scale reading for this line (used to get u)
    delta_u_rad    : angular separation of the doublet on the camera (rad)
    u_delta_u_rad  : uncertainty in delta_u_rad (rad)

    Returns the uncertainty in Delta_lambda, in nm.
    """
    u_rad = math.radians(get_u(reading_deg))
    u_u_rad = math.radians(get_u_uncertainty())

    d_du = -math.sin(u_rad) / (order * N_per_m) * delta_u_rad
    d_ddu = math.cos(u_rad) / (order * N_per_m)

    delta_lambda_m_uncertainty = math.sqrt((d_du * u_u_rad) ** 2 + (d_ddu * u_delta_u_rad) ** 2)
    return delta_lambda_m_uncertainty * 1e9


def resolving_power(lam_nm, delta_lam_nm):
    """
    R = lambda / Delta_lambda  (eq. Resolving_Power)

    lam_nm       : wavelength (nm)
    delta_lam_nm : smallest observable difference in wavelength (nm)

    Returns the (dimensionless) resolving power.
    """
    return lam_nm / delta_lam_nm

def theoretical_resolving_power(order, N_total_lines):
    """
    R_theor = m * N_tot  (eq. Theor_Resolving_Power)

    order          : diffraction order m
    N_total_lines  : total number of grating lines illuminated

    Returns the (dimensionless) theoretical resolving power.
    """
    return order * N_total_lines


def analyse_doublet(groups, group_lambda_bar):
    """
    Section "study the splitting of the yellow Hg lines" / "find splitting
    for all [Na] lines": use the camera pixel separation between the two
    yellow lines (doublet_pixels, set per-lamp by configure_lamp) to get
    Delta_u, then Delta_lambda (eq. deltalabda) and its uncertainty, then
    compare the measured and theoretical resolving power.

    Uses the order at which yellow1/yellow2 were actually measured (from
    the loaded data) and their average reading as the doublet's angle.

    groups            : dict mapping colour -> list of (order, reading_deg)
    group_lambda_bar  : dict colour -> weighted-average lambda (nm), from
                         compute_group_results, used for the doublet's mean
                         wavelength in R = lambda / Delta_lambda

    Returns the results dict, or None if this lamp has no yellow1/yellow2
    pair in the data to analyse.
    """
    if 'yellow1' not in groups or 'yellow2' not in groups:
        return None

    # Use the (order, reading) at which the doublet was measured. Both
    # lines are read at the same order; take the first reading for each
    # and their order (they should match -- see the sanity check below).
    order1, reading1 = groups['yellow1'][0]
    order2, reading2 = groups['yellow2'][0]
    assert order1 == order2, "yellow1/yellow2 were not measured at the same order"
    order = order1
    reading_deg = (reading1 + reading2) / 2  # average angle-scale reading for the doublet

    delta_u_rad = get_delta_u(doublet_pixels)
    u_delta_u_rad = get_delta_u_uncertainty()

    delta_lambda_nm = get_delta_lambda(order, reading_deg, delta_u_rad)
    u_delta_lambda_nm = get_delta_lambda_uncertainty(order, reading_deg, delta_u_rad, u_delta_u_rad)

    lam_mean_nm = (group_lambda_bar['yellow1'] + group_lambda_bar['yellow2']) / 2

    R_measured = resolving_power(lam_mean_nm, abs(delta_lambda_nm))
    R_theor = theoretical_resolving_power(abs(order), N_tot)

    print(f"\n--- {lamp}: yellow doublet splitting (order m = {order}) ---")
    print(f"pixel separation on camera   : {doublet_pixels} px")
    print(f"Delta_u                      : {math.degrees(delta_u_rad):.5f} +/- {math.degrees(u_delta_u_rad):.5f} deg")
    print(f"Delta_lambda                 : {delta_lambda_nm:.5f} +/- {u_delta_lambda_nm:.5f} nm")
    print(f"mean lambda (yellow1,2)      : {lam_mean_nm:.3f} nm")
    print(f"Measured resolving power R   : {R_measured:.0f}")
    print(f"Theoretical resolving power  : {R_theor:.0f}  (N_tot = {N_tot:.0f} lines, from {grating_width_mm} mm grating width)")

    return {
        'order': order,
        'delta_u_rad': delta_u_rad,
        'u_delta_u_rad': u_delta_u_rad,
        'delta_lambda_nm': delta_lambda_nm,
        'u_delta_lambda_nm': u_delta_lambda_nm,
        'lam_mean_nm': lam_mean_nm,
        'R_measured': R_measured,
        'R_theor': R_theor,
    }


def plot_wavelength_vs_order(groups, save_path=None, show=True):
    """
    Plot the individual lambda measurements (with error bars) against
    order m, one series per colour, with literature values overlaid as
    dotted horizontal lines.

    groups    : dict mapping colour -> list of (order, reading_deg) tuples
    save_path : if given, also save the figure here (in addition to
                displaying it, unless show=False)
    show      : if True (default), display the figure with plt.show();
                set False for headless runs where only the saved file
                is wanted
    """
    fig, ax = plt.subplots()

    for colour in sorted(groups):
        orders = [order for order, reading_deg in groups[colour]]
        lambdas = [get_wavelength(order, reading_deg) for order, reading_deg in groups[colour]]
        uncertainties = [get_wavelength_uncertainty(order, reading_deg)
                          for order, reading_deg in groups[colour]]

        line = ax.errorbar(orders, lambdas, yerr=uncertainties, fmt='o',
                            markersize=4, capsize=3, label=colour)
        if colour in LITERATURE_NM:
            ax.axhline(LITERATURE_NM[colour], linestyle=':', linewidth=1,
                        color=line[0].get_color())

    ax.set_xlabel('order $m$')
    ax.set_ylabel(r'$\lambda$ (nm)')
    _style_axes(ax)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()  # blocks until the figure window is closed, then continues
    else:
        plt.close(fig)

#################################################################################################################################################################
"""Main"""

def main():
    """
    Run the full analysis pipeline for both lamps in turn:
    configure lamp constants -> load data -> group by colour -> compute
    lambda per line (algebraic method) and the weighted average per colour
    -> check agreement with literature -> plot results -> analyse the
    yellow-doublet splitting and resolving power.
    """
    all_results = {}

    for lamp_name in ('Hg', 'Na'):
        configure_lamp(lamp_name)

        data = f'data_{lamp_name}.csv'
        runs = load_data(data)
        groups = group_by_colour(runs)
        group_lambda_bar, group_lambda_bar_unc = compute_group_results(groups)
        print()
        print_literature_agreement_table(group_lambda_bar, group_lambda_bar_unc)
        plot_wavelength_vs_order(groups, save_path=f'wavelength_vs_order_{lamp_name}.png')

        doublet_result = analyse_doublet(groups, group_lambda_bar)

        all_results[lamp_name] = {
            'group_lambda_bar': group_lambda_bar,
            'group_lambda_bar_unc': group_lambda_bar_unc,
            'doublet': doublet_result,
        }
        print()

    return all_results

if __name__ == '__main__':
    main()
