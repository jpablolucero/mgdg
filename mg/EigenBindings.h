#ifndef EIGENBINDINGS_H
#define EIGENBINDINGS_H

#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

template <std::size_t N,std::size_t M,typename number=double>
Eigen::Matrix<number,N,M> Eigenize(std::array<std::array<number,M>,N> data)
{
  Eigen::Matrix<number,N,M> eMatrix;
  eMatrix.setZero();
  for (int i = 0; i < data.size(); ++i)
    eMatrix.row(i) = Eigen::VectorXd::Map(&data[i][0], data[0].size());
  return eMatrix;
}

template <std::size_t M,typename number=double>
Eigen::Matrix<number,M,1> Eigenize(std::array<number,M> data)
{
  Eigen::Matrix<number,M,1> eMatrix(data.size(), 1);
  eMatrix.setZero();
  eMatrix.col(0) = Eigen::VectorXd::Map(&data[0], data.size());
  return eMatrix;
}

template <std::size_t N,std::size_t M,typename number=double>
Eigen::Matrix<number,N,M> Eigenize(const auto & data)
{
  Eigen::Matrix<number,N,M> A{};
  for (auto j = 0u ; j < N ; j++)
    {
      std::array<number,N> x{};
      x[j] = 1. ;
      const auto x1 = data*x;
      for (auto i = 0u ; i < N ; i ++)
      	{
      	  A(i,j) = x1[i] ;
      	}
    }
  return A;
}

#endif /* EIGENBINDINGS_H */
