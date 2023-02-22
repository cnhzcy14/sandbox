#include <iostream>
#include <Eigen/Dense>
#include <map>

// typedef Eigen::Matrix<float, 3, 4> Matrix3x4f_rm;

int main()
{
	// Declare and initialise a 3D matrix

	int w, h;
	w = 3;
	h = 4;
	Eigen::Matrix<float, 3, 4> p;
	p << 81, -4, -3, 0,
		-2, -1, 1, 0,
		2, 3, 4, 0;

	// Output the reduction operations
	std::cout << "p.sum(): " << p.sum() << std::endl;
	std::cout << "p.prod(): " << p.prod() << std::endl;
	std::cout << "p.mean(): " << p.mean() << std::endl;
	std::cout << "p.minCoeff(): " << p.minCoeff() << std::endl;
	std::cout << "p.maxCoeff(): " << p.maxCoeff() << std::endl;
	std::cout << "p.trace(): " << p.trace() << std::endl;
}