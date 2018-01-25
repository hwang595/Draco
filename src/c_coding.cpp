#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <complex>

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;

// we firstly make a quite naive example, take n, s and return polynomial a
Eigen::MatrixXcd solve_poly_a(int n, int s, Eigen::MatrixXcd R)
{
	int hat_s = 2 * s + 1;
	int p = 3;

	MatrixXcd C = MatrixXcd::Zero(n, n);
	MatrixXcd C1 = MatrixXcd::Zero(n, n-hat_s+1);
	MatrixXcd C2 = MatrixXcd::Zero(n, hat_s-1);
	MatrixXcd W_fake = MatrixXcd::Zero(n, p);
	MatrixXcd WPerp = MatrixXcd::Zero(hat_s-1, n);

	MatrixXcd A = MatrixXcd::Zero(s, s);
	MatrixXcd b = MatrixXcd::Zero(s, 1);

	MatrixXd G = MatrixXd::Random(p, 1);
	MatrixXd Q1 = MatrixXd::Random(n-2*s-1,p);
	MatrixXd Q = MatrixXd::Ones(n-2*s,p);
	MatrixXd eps = MatrixXd::Zero(n, 1);

	Q.topRows(n-2*s-1) = Q1;
	eps.topRows(s) = -100.0*MatrixXd::Ones(s, 1);

	double factor1 = 1/std::sqrt(n);
	// generate C matrix:
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{	
			if (j >= i)
			{
				if ((i == 0) || (j == 0))
				{
					C(i, j) = std::complex<double>(1, 0);
				}
				else{
					std::complex<double> temptElementValue = std::complex<double>(0, (-2*i*j*M_PI)/n);
					C(i, j) = std::exp(temptElementValue);
				} 
			}
			else{
				C(i, j) = C(j, i);
			}

		}
	}
	C *= factor1;
	// fetch C1 and C2:
	C1 = C.leftCols(n-hat_s+1);
	C2 = C.rightCols(hat_s-1);

	WPerp = C2.adjoint();
	W_fake = C1 * Q;

	//MatrixXcd R = W_fake * G + eps;
	// we assume here R is passed by Python side
	// and it is a complex vector, shape of R should
	// be n * 1
	MatrixXcd E2 = WPerp * R;

	// form A * x = b:
	for (int i = 0; i < s; ++i)
	{
		A.row(i) = E2.col(0).segment(2*s-i-s-1,s).transpose();
		b.row(i) = E2.row(2*s-i-1);
	}
	//MatrixXcd alpha = A.colPivHouseholderQr().solve(b);
	MatrixXcd alpha = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
	
	return alpha;
}

namespace py = pybind11;

PYBIND11_MODULE(c_coding, m)
{
  m.doc() = "pybind11 coding plugin";

  m.def("solve_poly_a", &solve_poly_a, py::arg("n"), py::arg("s"), py::arg("R"));
  //m.def("solve_poly_a", &solve_poly_a, py::arg("n"), py::arg("s"));
}