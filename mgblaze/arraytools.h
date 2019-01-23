#ifndef ARRAYTOOLS_H
#define ARRAYTOOLS_H

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <blaze/Math.h>


template <std::size_t Nb, std::size_t Ns=0, std::size_t N, typename number=double>
constexpr auto block_jacobi(const blaze::StaticMatrix<number,N,N>& A)
{
  blaze::StaticMatrix<number,N,N> J{} ;

  if constexpr (Ns != 0)
		 {
		   blaze::StaticMatrix<number,Ns,Ns> B0{} ;
		   for (auto i = 0u ; i < Ns ; ++i)
		     for (auto j = 0u ; j < Ns ; ++j)
		       B0(i,j) = A(i,j) ;
		   invert(B0);
		   for (auto i = 0u ; i < Ns ; ++i)
		     for (auto j = 0u ; j < Ns ; ++j)
		       J(i,j) = B0(i,j) ;
		 }
  
  blaze::StaticMatrix<number,Nb,Nb> B1{} ;
  for (auto b = 0u ; b < N / Nb - 1; ++b)
    {
      for (auto i = Ns ; i < Ns + Nb ; ++i)
	for (auto j = Ns ; j < Ns + Nb ; ++j)
	  B1(i - Ns,j - Ns) = A(b*Nb+i,b*Nb+j) ;
      invert(B1);
      for (auto i = Ns ; i < Ns + Nb ; ++i)
	for (auto j = Ns ; j < Ns + Nb ; ++j)
	  J(b*Nb+i,b*Nb+j) = B1(i - Ns,j - Ns) ;
    }

  blaze::StaticMatrix<number,Nb-Ns,Nb-Ns> B2{} ;
  for (auto i = N - (Nb - Ns) ; i < N ; ++i)
    for (auto j = N - (Nb - Ns) ; j < N ; ++j)
      B2(i - (N - (Nb - Ns)),j - (N - (Nb - Ns))) = A(i,j) ;
  invert(B2);
  for (auto i = N - (Nb - Ns) ; i < N ; ++i)
    for (auto j = N - (Nb - Ns) ; j < N ; ++j)
      J(i,j) = B2(i - (N - (Nb - Ns)),j - (N - (Nb - Ns))) ;

  return J ;
}

#endif /* ARRAYTOOLS_H */
