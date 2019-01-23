#ifndef ARRAYTOOLS_H
#define ARRAYTOOLS_H

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <Eigen/Dense>

template <int Nb, int Ns=0, int N, typename number=double>
constexpr auto block_jacobi(const Eigen::Matrix<number,N,N>& A)
{
  Eigen::Matrix<number,N,N> J{} ;
  J.setZero() ;

  if constexpr (Ns != 0)
		 {
		   Eigen::Matrix<number,Ns,Ns> B0{} ;
		   B0.setZero() ;
		   for (auto i = 0u ; i < Ns ; ++i)
		     for (auto j = 0u ; j < Ns ; ++j)
		       B0(i,j) = A(i,j) ;
		   const auto tmp = B0.inverse() ;
		   for (auto i = 0u ; i < Ns ; ++i)
		     for (auto j = 0u ; j < Ns ; ++j)
		       J(i,j) = tmp(i,j) ;
		 }
  
  Eigen::Matrix<number,Nb,Nb> B1{} ;
  B1.setZero() ;
  for (auto b = 0u ; b < N / Nb - 1; ++b)
    {
      for (auto i = Ns ; i < Ns + Nb ; ++i)
	for (auto j = Ns ; j < Ns + Nb ; ++j)
	  B1(i - Ns,j - Ns) = A(b*Nb+i,b*Nb+j) ;
      const auto tmp = B1.inverse() ;
      for (auto i = Ns ; i < Ns + Nb ; ++i)
	for (auto j = Ns ; j < Ns + Nb ; ++j)
	  J(b*Nb+i,b*Nb+j) = tmp(i - Ns,j - Ns) ;
    }

  Eigen::Matrix<number,Nb-Ns,Nb-Ns> B2{} ;
  B2.setZero() ;
  for (auto i = N - (Nb - Ns) ; i < N ; ++i)
    for (auto j = N - (Nb - Ns) ; j < N ; ++j)
      B2(i - (N - (Nb - Ns)),j - (N - (Nb - Ns))) = A(i,j) ;
  const auto tmp = B2.inverse();
  for (auto i = N - (Nb - Ns) ; i < N ; ++i)
    for (auto j = N - (Nb - Ns) ; j < N ; ++j)
      J(i,j) = tmp(i - (N - (Nb - Ns)),j - (N - (Nb - Ns))) ;

  return J ;
}

#endif /* ARRAYTOOLS_H */
