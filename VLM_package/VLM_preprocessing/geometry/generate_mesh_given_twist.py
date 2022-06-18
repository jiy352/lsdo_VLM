import csdl
import csdl_om

import numpy as np
from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import *


class Rotate(csdl.CustomExplicitOperation):
    """
    OpenMDAO component that manipulates the mesh by compute rotation matrices given mesh and
    rotation angles in degrees.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    theta_y[ny] : numpy array
        1-D array of rotation angles about y-axis for each wing slice in degrees.
    symmetry : boolean
        Flag set to True if surface is reflected about y=0 plane.
    rotate_x : boolean
        Flag set to True if the user desires the twist variable to always be
        applied perpendicular to the wing (say, in the case of a winglet).

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the twisted aerodynamic surface.
    """
    def initialize(self):
        """
        Declare parameters.
        """
        self.parameters.declare("val", desc="Initial value for dihedral.")
        self.parameters.declare("mesh_shape",
                                desc="Tuple containing mesh shape (nx, ny).")
        self.parameters.declare(
            "symmetry",
            default=False,
            desc="Flag set to true if surface is reflected about y=0 plane.")
        self.parameters.declare(
            "rotate_x",
            default=True,
            desc="Flag set to True if the user desires the twist variable to "
            "always be applied perpendicular to the wing (say, in the case of "
            "a winglet).",
        )

    def define(self):
        mesh_shape = self.parameters["mesh_shape"]
        val = self.parameters["val"]

        self.add_input("twist", val=val)
        self.add_input("in_mesh", shape=mesh_shape, units="m")

        self.add_output("mesh", shape=mesh_shape, units="m")

        nx, ny, _ = mesh_shape
        nn = nx * ny * 3
        rows = np.arange(nn)
        col = np.tile(np.zeros(3), ny) + np.repeat(np.arange(ny), 3)
        cols = np.tile(col, nx)

        self.declare_derivatives("mesh", "twist", rows=rows, cols=cols)

        row_base = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        col_base = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

        # Diagonal
        nn = nx * ny
        dg_row = np.tile(row_base, nn) + np.repeat(3 * np.arange(nn), 9)
        dg_col = np.tile(col_base, nn) + np.repeat(3 * np.arange(nn), 9)

        # Leading and Trailing edge on diagonal terms.
        row_base_y = np.tile(row_base, ny) + np.repeat(3 * np.arange(ny), 9)
        col_base_y = np.tile(col_base, ny) + np.repeat(3 * np.arange(ny), 9)
        nn2 = 3 * ny
        te_dg_row = np.tile(row_base_y, nx - 1) + np.repeat(
            nn2 * np.arange(nx - 1), 9 * ny)
        le_dg_col = np.tile(col_base_y, nx - 1)
        le_dg_row = te_dg_row + nn2
        te_dg_col = le_dg_col + 3 * ny * (nx - 1)

        # Leading and Trailing edge off diagonal terms.
        if self.parameters["symmetry"]:
            row_base_y = np.tile(row_base, ny - 1) + np.repeat(
                3 * np.arange(ny - 1), 9)
            col_base_y = np.tile(col_base + 3, ny - 1) + np.repeat(
                3 * np.arange(ny - 1), 9)

            nn2 = 3 * ny
            te_od_row = np.tile(row_base_y, nx) + np.repeat(
                nn2 * np.arange(nx), 9 * (ny - 1))
            le_od_col = np.tile(col_base_y, nx)
            te_od_col = le_od_col + 3 * ny * (nx - 1)

            rows = np.concatenate(
                [dg_row, le_dg_row, te_dg_row, te_od_row, te_od_row])
            cols = np.concatenate(
                [dg_col, le_dg_col, te_dg_col, le_od_col, te_od_col])

        else:
            n_sym = (ny - 1) // 2

            row_base_y1 = np.tile(row_base, n_sym) + np.repeat(
                3 * np.arange(n_sym), 9)
            col_base_y1 = np.tile(col_base + 3, n_sym) + np.repeat(
                3 * np.arange(n_sym), 9)

            row_base_y2 = row_base_y1 + 3 * n_sym + 3
            col_base_y2 = col_base_y1 + 3 * n_sym - 3

            nn2 = 3 * ny

            te_od_row1 = np.tile(row_base_y1, nx) + np.repeat(
                nn2 * np.arange(nx), 9 * n_sym)
            le_od_col1 = np.tile(col_base_y1, nx)
            te_od_col1 = le_od_col1 + 3 * ny * (nx - 1)
            te_od_row2 = np.tile(row_base_y2, nx) + np.repeat(
                nn2 * np.arange(nx), 9 * n_sym)
            le_od_col2 = np.tile(col_base_y2, nx)
            te_od_col2 = le_od_col2 + 3 * ny * (nx - 1)

            rows = np.concatenate([
                dg_row, le_dg_row, te_dg_row, te_od_row1, te_od_row2,
                te_od_row1, te_od_row2
            ])
            cols = np.concatenate([
                dg_col, le_dg_col, te_dg_col, le_od_col1, le_od_col2,
                te_od_col1, te_od_col2
            ])

        self.declare_derivatives("mesh", "in_mesh", rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        symmetry = self.parameters["symmetry"]
        rotate_x = self.parameters["rotate_x"]
        theta_y = inputs["twist"]
        mesh = inputs["in_mesh"]

        te = mesh[-1]
        le = mesh[0]
        quarter_chord = 0.25 * te + 0.75 * le

        _, ny, _ = mesh.shape

        if rotate_x:
            # Compute spanwise z displacements along quarter chord
            if symmetry:
                dz_qc = quarter_chord[:-1, 2] - quarter_chord[1:, 2]
                dy_qc = quarter_chord[:-1, 1] - quarter_chord[1:, 1]
                theta_x = np.arctan(dz_qc / dy_qc)

                # Prepend with 0 so that root is not rotated
                rad_theta_x = np.append(theta_x, 0.0)
            else:
                root_index = int((ny - 1) / 2)
                dz_qc_left = quarter_chord[:root_index,
                                           2] - quarter_chord[1:root_index + 1,
                                                              2]
                dy_qc_left = quarter_chord[:root_index,
                                           1] - quarter_chord[1:root_index + 1,
                                                              1]
                theta_x_left = np.arctan(dz_qc_left / dy_qc_left)
                dz_qc_right = quarter_chord[root_index + 1:,
                                            2] - quarter_chord[root_index:-1,
                                                               2]
                dy_qc_right = quarter_chord[root_index + 1:,
                                            1] - quarter_chord[root_index:-1,
                                                               1]
                theta_x_right = np.arctan(dz_qc_right / dy_qc_right)

                # Concatenate thetas
                rad_theta_x = np.concatenate(
                    (theta_x_left, np.zeros(1), theta_x_right))

        else:
            rad_theta_x = 0.0

        rad_theta_y = theta_y * np.pi / 180.0

        mats = np.zeros((ny, 3, 3), dtype=type(rad_theta_y[0]))

        cos_rtx = np.cos(rad_theta_x)
        cos_rty = np.cos(rad_theta_y)
        sin_rtx = np.sin(rad_theta_x)
        sin_rty = np.sin(rad_theta_y)

        mats[:, 0, 0] = cos_rty
        mats[:, 0, 2] = sin_rty
        mats[:, 1, 0] = sin_rtx * sin_rty
        mats[:, 1, 1] = cos_rtx
        mats[:, 1, 2] = -sin_rtx * cos_rty
        mats[:, 2, 0] = -cos_rtx * sin_rty
        mats[:, 2, 1] = sin_rtx
        mats[:, 2, 2] = cos_rtx * cos_rty

        outputs["mesh"] = np.einsum("ikj, mij -> mik", mats,
                                    mesh - quarter_chord) + quarter_chord

    def compute_derivatives(self, inputs, derivatives):
        symmetry = self.parameters["symmetry"]
        rotate_x = self.parameters["rotate_x"]
        theta_y = inputs["twist"]
        mesh = inputs["in_mesh"]

        te = mesh[-1]
        le = mesh[0]
        quarter_chord = 0.25 * te + 0.75 * le

        nx, ny, _ = mesh.shape

        if rotate_x:
            # Compute spanwise z displacements along quarter chord
            if symmetry:
                dz_qc = quarter_chord[:-1, 2] - quarter_chord[1:, 2]
                dy_qc = quarter_chord[:-1, 1] - quarter_chord[1:, 1]
                theta_x = np.arctan(dz_qc / dy_qc)

                # Prepend with 0 so that root is not rotated
                rad_theta_x = np.append(theta_x, 0.0)

                fact = 1.0 / (1.0 + (dz_qc / dy_qc)**2)

                dthx_dq = np.zeros((ny, 3))
                dthx_dq[:-1, 1] = -dz_qc * fact / dy_qc**2
                dthx_dq[:-1, 2] = fact / dy_qc

            else:
                root_index = int((ny - 1) / 2)
                dz_qc_left = quarter_chord[:root_index,
                                           2] - quarter_chord[1:root_index + 1,
                                                              2]
                dy_qc_left = quarter_chord[:root_index,
                                           1] - quarter_chord[1:root_index + 1,
                                                              1]
                theta_x_left = np.arctan(dz_qc_left / dy_qc_left)
                dz_qc_right = quarter_chord[root_index + 1:,
                                            2] - quarter_chord[root_index:-1,
                                                               2]
                dy_qc_right = quarter_chord[root_index + 1:,
                                            1] - quarter_chord[root_index:-1,
                                                               1]
                theta_x_right = np.arctan(dz_qc_right / dy_qc_right)

                # Concatenate thetas
                rad_theta_x = np.concatenate(
                    (theta_x_left, np.zeros(1), theta_x_right))

                fact_left = 1.0 / (1.0 + (dz_qc_left / dy_qc_left)**2)
                fact_right = 1.0 / (1.0 + (dz_qc_right / dy_qc_right)**2)

                dthx_dq = np.zeros((ny, 3))
                dthx_dq[:root_index,
                        1] = -dz_qc_left * fact_left / dy_qc_left**2
                dthx_dq[root_index + 1:,
                        1] = -dz_qc_right * fact_right / dy_qc_right**2
                dthx_dq[:root_index, 2] = fact_left / dy_qc_left
                dthx_dq[root_index + 1:, 2] = fact_right / dy_qc_right

        else:
            rad_theta_x = 0.0

        deg2rad = np.pi / 180.0
        rad_theta_y = theta_y * deg2rad

        mats = np.zeros((ny, 3, 3), dtype=type(rad_theta_y[0]))

        cos_rtx = np.cos(rad_theta_x)
        cos_rty = np.cos(rad_theta_y)
        sin_rtx = np.sin(rad_theta_x)
        sin_rty = np.sin(rad_theta_y)

        mats[:, 0, 0] = cos_rty
        mats[:, 0, 2] = sin_rty
        mats[:, 1, 0] = sin_rtx * sin_rty
        mats[:, 1, 1] = cos_rtx
        mats[:, 1, 2] = -sin_rtx * cos_rty
        mats[:, 2, 0] = -cos_rtx * sin_rty
        mats[:, 2, 1] = sin_rtx
        mats[:, 2, 2] = cos_rtx * cos_rty

        dmats_dthy = np.zeros((ny, 3, 3))
        dmats_dthy[:, 0, 0] = -sin_rty * deg2rad
        dmats_dthy[:, 0, 2] = cos_rty * deg2rad
        dmats_dthy[:, 1, 0] = sin_rtx * cos_rty * deg2rad
        dmats_dthy[:, 1, 2] = sin_rtx * sin_rty * deg2rad
        dmats_dthy[:, 2, 0] = -cos_rtx * cos_rty * deg2rad
        dmats_dthy[:, 2, 2] = -cos_rtx * sin_rty * deg2rad

        d_dthetay = np.einsum("ikj, mij -> mik", dmats_dthy,
                              mesh - quarter_chord)
        derivatives["mesh", "twist"] = d_dthetay.flatten()

        nn = nx * ny * 9
        derivatives["mesh", "in_mesh"][:nn] = np.tile(mats.flatten(), nx)

        # Quarter chord direct contribution.
        eye = np.tile(np.eye(3).flatten(), ny).reshape(ny, 3, 3)
        d_qch = (eye - mats).flatten()

        nqc = ny * 9
        derivatives["mesh", "in_mesh"][:nqc] += 0.75 * d_qch
        derivatives["mesh", "in_mesh"][nn - nqc:nn] += 0.25 * d_qch

        if rotate_x:

            dmats_dthx = np.zeros((ny, 3, 3))
            dmats_dthx[:, 1, 0] = cos_rtx * sin_rty
            dmats_dthx[:, 1, 1] = -sin_rtx
            dmats_dthx[:, 1, 2] = -cos_rtx * cos_rty
            dmats_dthx[:, 2, 0] = sin_rtx * sin_rty
            dmats_dthx[:, 2, 1] = cos_rtx
            dmats_dthx[:, 2, 2] = -sin_rtx * cos_rty

            d_dthetax = np.einsum("ikj, mij -> mik", dmats_dthx,
                                  mesh - quarter_chord)
            d_dq = np.einsum("ijk, jm -> ijkm", d_dthetax, dthx_dq)

            d_dq_flat = d_dq.flatten()

            del_n = nn - 9 * ny
            nn2 = nn + del_n
            nn3 = nn2 + del_n
            derivatives["mesh", "in_mesh"][nn:nn2] = 0.75 * d_dq_flat[-del_n:]
            derivatives["mesh", "in_mesh"][nn2:nn3] = 0.25 * d_dq_flat[:del_n]

            # Contribution back to main diagonal.
            del_n = 9 * ny
            derivatives["mesh", "in_mesh"][:nqc] += 0.75 * d_dq_flat[:del_n]
            derivatives["mesh",
                        "in_mesh"][nn - nqc:nn] += 0.25 * d_dq_flat[-del_n:]

            # Quarter chord direct contribution.
            d_qch_od = np.tile(d_qch.flatten(), nx - 1)
            derivatives["mesh", "in_mesh"][nn:nn2] += 0.75 * d_qch_od
            derivatives["mesh", "in_mesh"][nn2:nn3] += 0.25 * d_qch_od

            # off-off diagonal pieces
            if symmetry:
                d_dq_flat = d_dq[:, :-1, :, :].flatten()

                del_n = nn - 9 * nx
                nn4 = nn3 + del_n
                derivatives["mesh", "in_mesh"][nn3:nn4] = -0.75 * d_dq_flat
                nn5 = nn4 + del_n
                derivatives["mesh", "in_mesh"][nn4:nn5] = -0.25 * d_dq_flat

            else:
                d_dq_flat1 = d_dq[:, :root_index, :, :].flatten()
                d_dq_flat2 = d_dq[:, root_index + 1:, :, :].flatten()

                del_n = nx * root_index * 9
                nn4 = nn3 + del_n
                derivatives["mesh", "in_mesh"][nn3:nn4] = -0.75 * d_dq_flat1
                nn5 = nn4 + del_n
                derivatives["mesh", "in_mesh"][nn4:nn5] = -0.75 * d_dq_flat2
                nn6 = nn5 + del_n
                derivatives["mesh", "in_mesh"][nn5:nn6] = -0.25 * d_dq_flat1
                nn7 = nn6 + del_n
                derivatives["mesh", "in_mesh"][nn6:nn7] = -0.25 * d_dq_flat2


# NY = 7
# NX = 5

# symmetry = False
# mesh = generate_simple_mesh(NX, NY)

# val = np.zeros(NY)
# top_model = csdl.Model()

# in_mesh = top_model.declare_variable('in_mesh', val=mesh)
# twist = top_model.declare_variable('twist', val=np.random.random(NY))
# product = csdl.custom(twist,
#                       in_mesh,
#                       op=Rotate(val=val,
#                                 mesh_shape=mesh.shape,
#                                 symmetry=symmetry))
# top_model.register_output('product', product)
# sim = csdl_om.Simulator(top_model)

# sim.run()

# sim.prob.check_partials(compact_print=True, abs_err_tol=1e-5, rel_err_tol=1e-5)
