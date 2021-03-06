#= -------------------------------------------------------------------------
# @file test.py
#
# @date 03/25/16 19:13:57
# @author Martin Noblia
# @email martin.noblia@openmailbox.org
#
# @brief
#
# @detail
#
#  Licence:
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License

#---------------------------------------------------------------------------=#

import unittest
import numpy as np
from numeric_methods_module import*

class MyTest(unittest.TestCase):

    def test_backward_subs(self):

        n = 4
        A = np.random.random((n,n))
        U = np.triu(A)
        x = np.ones(n)
        b = U @ x
        self.assertEqual(np.allclose(backward_subs(U, b), x), True)

    def test_forward_subs(self):

        n = 4
        A = np.random.random((n,n))
        L = np.tril(A)
        x = np.ones(n)
        b = L @ x

        self.assertEqual(np.allclose(forward_subs(L, b), x), True)

    def test_lu_fact(self):

        n = 4
        A = np.random.random((n,n))
        B = np.copy(A)
        L, U = lu_fact(A)
        self.assertEqual(np.allclose((L @ U - B), np.zeros_like(B)), True)


unittest.main()


