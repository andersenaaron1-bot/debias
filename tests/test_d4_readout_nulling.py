import unittest

import torch

from aisafety.mech.readout_nulling import orthonormal_basis, project_out


class D4ReadoutNullingTests(unittest.TestCase):
    def test_project_out_removes_direction_and_preserves_orthogonal_signal(self) -> None:
        basis = orthonormal_basis(torch.tensor([[1.0, 0.0, 0.0]]))
        pooled = torch.tensor([[3.0, 2.0, -1.0], [-4.0, 0.5, 2.0]])
        nulled = project_out(pooled, basis)
        expected = torch.tensor([[0.0, 2.0, -1.0], [0.0, 0.5, 2.0]])
        self.assertTrue(torch.allclose(nulled, expected, atol=1e-6))

    def test_orthonormal_basis_drops_zero_rows(self) -> None:
        basis = orthonormal_basis(torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]))
        self.assertEqual(tuple(basis.shape), (2, 1))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(basis[:, 0])), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
