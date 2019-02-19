#ifndef ARRAYTOOLS_H
#define ARRAYTOOLS_H

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>

template <std::size_t nr, typename number=double>
constexpr typename std::enable_if<std::is_arithmetic<number>::value,std::array<number,nr> >::type
operator *(const number x, const std::array<number,nr> & y)
{
  std::array<number,nr> res{};
  int par = (nr >= 128) ? 1 : 0 ;
  #ifdef PARALLEL
  #pragma omp parallel for if(par) default(none) shared(y,res)
  #endif
  for (auto i=0u;i<nr;++i)
    res[i] = x*y[i];
  return res ;
}

template <std::size_t nr, typename number=double>
constexpr typename std::enable_if<std::is_arithmetic<number>::value,number>::type
operator *(const std::array<number,nr> & x, const std::array<number,nr> & y)
{
  number res = 0.;
  int par = (nr >= 128) ? 1 : 0 ;
  #ifdef PARALLEL
  #pragma omp parallel for if(par) reduction(+:res) default(none) shared(x,y)
  #endif
  for (auto i=0u;i<x.size();++i)
    res += x[i]*y[i];
  return res ;
}

template <std::size_t nr, typename number=double>
constexpr typename std::enable_if<std::is_arithmetic<number>::value,std::array<number,nr> >::type
operator -(const std::array<number,nr> & x, const std::array<number,nr> & y)
{
  std::array<number,nr> res{};
  int par = (nr >= 128) ? 1 : 0 ;
  #ifdef PARALLEL
  #pragma omp parallel for if(par) default(none) shared(x,y,res)
  #endif
  for (auto i=0u;i<x.size();++i)
    res[i] = x[i] - y[i];
  return res ;
}

template <std::size_t nr, typename number=double>
constexpr typename std::enable_if<std::is_arithmetic<number>::value,std::array<number,nr> >::type
operator +(const std::array<number,nr> & x, const std::array<number,nr> & y)
{
  std::array<number,nr> res{};
  int par = (nr >= 128) ? 1 : 0 ;
  #ifdef PARALLEL
  #pragma omp parallel for if(par) default(none) shared(x,y,res)
  #endif
  for (auto i=0u;i<nr;++i)
    res[i] = x[i] + y[i];
  return res ;
}

template <std::size_t nr, std::size_t nc, typename number=double>
constexpr typename std::enable_if<std::is_arithmetic<number>::value,std::array<number,nr> >::type
operator *(const std::array<std::array<number,nc>,nr> & A,const std::array<number,nc> & x)
{
  std::array<number,nr> res{} ;
  int par = (nr >= 128) ? 1 : 0 ;
  #ifdef PARALLEL
  #pragma omp parallel for if(par) default(none) shared(A,x,res)
  #endif
  for (auto i=0u;i<nr;i++)
    for (auto j=0u;j<nc;j++)
      res[i] += A[i][j]*x[j];
  return res ;
}

template <std::size_t nrA, std::size_t ncA, std::size_t nrB, std::size_t ncB, typename number=double>
constexpr typename std::enable_if<std::is_arithmetic<number>::value,
				  std::array<std::array<number,ncB>,nrA> >::type
operator *(const std::array<std::array<number,ncA>,nrA> & A,
	   const std::array<std::array<number,ncB>,nrB> & B)
{
  std::array<std::array<number,ncB>,nrA> res{} ;
  int par = ((nrA >= 128) or (ncA >= 128) or (nrB >= 128) or (ncB >= 128)) ? 1 : 0 ;
  #ifdef PARALLEL
  #pragma omp parallel for if(par) default(none) shared(A,B,res)
  #endif
  for (auto k=0u;k<ncB;++k)
    for (auto i=0u;i<nrA;++i)
      for (auto j=0u;j<nrB;++j)      
	res[i][k] += A[i][j]*B[j][k];
  return res ;
}

template <std::size_t nrA, std::size_t ncA, typename number=double>
constexpr typename std::enable_if<std::is_arithmetic<number>::value,
				  std::array<std::array<number,ncA>,nrA> >::type
operator -(const std::array<std::array<number,ncA>,nrA> & A,
	   const std::array<std::array<number,ncA>,nrA> & B)
{
  std::array<std::array<number,ncA>,nrA> res{} ;
  int par = ((nrA >= 128) or (ncA >= 128)) ? 1 : 0 ;
  #ifdef PARALLEL
  #pragma omp parallel for if(par) default(none) shared(A,B,res)
  #endif
  for (auto i=0u;i<nrA;++i)
    for (auto j=0u;j<nrA;++j)      
      res[i][j] = A[i][j] - B[i][j];
  return res ;
}

template <std::size_t nrA, std::size_t ncA, typename number=double>
constexpr typename std::enable_if<std::is_arithmetic<number>::value,
				  std::array<std::array<number,ncA>,nrA> >::type
operator +(const std::array<std::array<number,ncA>,nrA> & A,
	   const std::array<std::array<number,ncA>,nrA> & B)
{
  std::array<std::array<number,ncA>,nrA> res{} ;
  int par = ((nrA >= 128) or (ncA >= 128)) ? 1 : 0 ;
  #ifdef PARALLEL
  #pragma omp parallel for if(par) default(none) shared(A,B,res)
  #endif
  for (auto i=0u;i<nrA;++i)
    for (auto j=0u;j<nrA;++j)      
      res[i][j] = A[i][j] + B[i][j];
  return res ;
}

template <std::size_t nrA, std::size_t ncA, typename number=double>
constexpr typename std::enable_if<std::is_arithmetic<number>::value,
				  std::array<std::array<number,ncA>,nrA> >::type
operator *(const std::array<std::array<number,ncA>,nrA> & A,
	   const number B)
{
  std::array<std::array<number,ncA>,nrA> res{} ;
  int par = ((nrA >= 128) or (ncA >= 128)) ? 1 : 0 ;
  #ifdef PARALLEL
  #pragma omp parallel for if(par) default(none) shared(A,res)
  #endif
  for (auto i=0u;i<nrA;++i)
    for (auto j=0u;j<ncA;++j)      
	res[i][j] = A[i][j] * B;
  return res ;
}

template <std::size_t nrB, std::size_t ncB, typename number=double>
constexpr typename std::enable_if<std::is_arithmetic<number>::value,
				  std::array<std::array<number,ncB>,nrB> >::type
operator *(const number A,const std::array<std::array<number,ncB>,nrB> & B)
{
  std::array<std::array<number,ncB>,nrB> res{} ;
  int par = ((nrB >= 128) or (ncB >= 128)) ? 1 : 0 ;
  #ifdef PARALLEL
  #pragma omp parallel for if(par) default(none) shared(B,res)
  #endif
  for (auto i=0u;i<nrB;++i)
    for (auto j=0u;j<nrB;++j)      
      res[i][j] = A * B[i][j];
  return res ;
}

constexpr auto operator *(const auto & A,const auto & x)
{
  return A(x) ;
}


template <std::size_t nr, std::size_t nc, typename number=double>
constexpr auto print(const std::array<std::array<number,nc>,nr> & A)
{
  for (auto i=0u;i<nr;++i)
    {
      for (auto j=0u;j<nc;++j)
  	std::cout << std::right << std::setw(12) << A[i][j];
      std::cout << std::endl ;
    }
  std::cout << std::endl ;
}

template <std::size_t n, typename number=double>
constexpr auto print(const std::array<number,n> & v)
{
  for (const auto el : const_cast<std::array<number,n>& >(v))
    std::cout << el << " " ;
  std::cout << std::endl ;
}

template <std::size_t n, typename number=double>
constexpr auto print(const volatile std::array<number,n> & v)
{
  for (const auto el : const_cast<std::array<number,n>& >(v))
    std::cout << el << " " ;
  std::cout << std::endl ;
}

template <typename number=double>
constexpr auto invert(const std::array<std::array<number, 1>,1>& x);

template <typename number=double>
constexpr auto invert(const std::array<std::array<number,2>,2>& a) ;

template <std::size_t N, typename number=double>
constexpr auto invert(const std::array<std::array<number,N>,N>& a) ;

template <typename AA, typename BB, typename CC, typename DD>
constexpr auto block_inverse(const AA& A, const BB& B, const CC& C,
			     const DD& D) {
  const auto invA = invert(A);
  const auto invS = invert(D - C * invA * B);

  const auto Q = invA + invA * B * invS * C * invA;
  const auto R = -1. * invA * B * invS;
  const auto S = -1. * invS * C * invA;
  const auto T = invS;

  return std::make_tuple(Q, R, S, T);
}

template <typename number=double>
constexpr auto invert(const std::array<std::array<number, 1>,1>& x)
{
  return std::array<std::array<number,1>,1>{std::array<number,1>{1./x[0][0]}};
}

template <typename number=double>
constexpr auto invert(const std::array<std::array<number,2>,2>& a) {
  const auto A = std::array<std::array<number,1>,1>{std::array<number,1>{a[0][0]}} ;
  const auto B = std::array<std::array<number,1>,1>{std::array<number,1>{a[0][1]}} ;
  const auto C = std::array<std::array<number,1>,1>{std::array<number,1>{a[1][0]}} ;
  const auto D = std::array<std::array<number,1>,1>{std::array<number,1>{a[1][1]}} ;

  const auto [Q, R, S, T] = block_inverse(A, B, C, D);

  return std::array<std::array<number,2>,2>{std::array<number,2>{Q[0][0],R[0][0]},
      std::array<number,2>{S[0][0],T[0][0]}};
}

template <std::size_t N, typename number=double>
constexpr auto invert(const std::array<std::array<number,N>,N>& a) {
  std::array<std::array<number,N / 2>,N / 2> A{} ;
  std::array<std::array<number,N - N / 2>,N / 2> B{} ;
  std::array<std::array<number,N / 2>,N - N / 2> C{} ;
  std::array<std::array<number,N - N / 2>,N - N / 2> D{} ;

  for (auto i=0u;i<N / 2;++i)
    for (auto j=0u;j<N / 2;++j)
      A[i][j] = a[i][j] ;
  for (auto i=0u;i<N / 2;++i)
    for (auto j=0u;j < N - N / 2;++j)
      B[i][j] = a[i][j+N / 2] ;
  for (auto i=0u;i < N - N/2;++i)
    for (auto j=0u;j < N / 2;++j)
      C[i][j] = a[i+N/2][j] ;
  for (auto i=0u;i < N - N/2;++i)
    for (auto j=0u;j < N - N/2;++j)
      D[i][j] = a[i+N / 2][j+N / 2] ;

  const auto [Q, R, S, T] = block_inverse(A, B, C, D);

  std::array<std::array<number,N>,N> res{} ;

  for (auto i=0u;i<N / 2;++i)
    for (auto j=0u;j<N / 2;++j)
      res[i][j] = Q[i][j] ;
  for (auto i=0u;i<N / 2;++i)
    for (auto j=0u;j < N - N / 2;++j)
      res[i][j+N / 2] = R[i][j] ;
  for (auto i=0u;i < N - N/2;++i)
    for (auto j=0u;j < N / 2;++j)
      res[i+N / 2][j] = S[i][j] ;
  for (auto i=0u;i < N - N/2;++i)
    for (auto j=0u;j < N - N / 2;++j)
      res[i+N / 2][j+N / 2] = T[i][j] ;
  /*
  std::array<std::array<number,N>,N> I{} ;
  for (auto i=0u;i < N;++i)
    I[i][i] = 1.;

  const auto diff = a*res - I ;
  double norm{} ;
  for (auto i=0u;i < N;++i)
    for (auto j=0u;j < N;++j)
      norm += std::abs(diff[i][j]);

  if (norm > 1.E-10) std::exit(1) ;
  */
  return res;
}

template <std::size_t Nb, std::size_t Ns=0, std::size_t N, typename number=double>
constexpr auto block_jacobi(const std::array<std::array<number,N>,N>& A)
{
  if (Nb == N) return invert(A);
  
  std::array<std::array<number,N>,N> J{} ;
  
  if constexpr (Ns != 0)
		 {
		   std::array<std::array<number,Ns>,Ns> B0{} ;
		   for (auto i = 0u ; i < Ns ; ++i)
		     for (auto j = 0u ; j < Ns ; ++j)
		       B0[i][j] = A[i][j] ;
		   B0 = invert(B0);
		   for (auto i = 0u ; i < Ns ; ++i)
		     for (auto j = 0u ; j < Ns ; ++j)
		       J[i][j] = B0[i][j] ;
		 }

  std::array<std::array<number,Nb>,Nb> B1{} ;
  for (auto i = Ns ; i < Ns + Nb ; ++i)
    for (auto j = Ns ; j < Ns + Nb ; ++j)
      B1[i - Ns][j - Ns] = A[i][j] ;
  B1 = invert(B1);
  for (auto i = Ns ; i < Ns + Nb ; ++i)
    for (auto j = Ns ; j < Ns + Nb ; ++j)
      J[i][j] = B1[i - Ns][j - Ns] ;

  std::array<std::array<number,Nb>,Nb> B2{} ;
  for (auto i = Ns ; i < Ns + Nb ; ++i)
    for (auto j = Ns ; j < Ns + Nb ; ++j)
      B2[i - Ns][j - Ns] = A[Nb+i][Nb+j] ;
  B2 = invert(B2);
  
  int par = (N >= 128) ? 1 : 0 ;
  #ifdef PARALLEL
  #pragma omp parallel for if(par) default(none) shared(J,B2)
  #endif
  for (auto b = 1u ; b < (N / Nb - 2) ; ++b)
    for (auto i = Ns ; i < Ns + Nb ; ++i)
      for (auto j = Ns ; j < Ns + Nb ; ++j)
	J[b*Nb+i][b*Nb+j] = B2[i - Ns][j - Ns] ;

  std::array<std::array<number,Nb>,Nb> B3{} ;
  for (auto i = Ns ; i < Ns + Nb ; ++i)
    for (auto j = Ns ; j < Ns + Nb ; ++j)
      B3[i - Ns][j - Ns] = A[(N / Nb - 2)*Nb+i][(N / Nb - 2)*Nb+j] ;
  B3 = invert(B3);
  for (auto i = Ns ; i < Ns + Nb ; ++i)
    for (auto j = Ns ; j < Ns + Nb ; ++j)
      J[(N / Nb - 2)*Nb+i][(N / Nb - 2)*Nb+j] = B3[i - Ns][j - Ns] ;

  std::array<std::array<number,Nb-Ns>,Nb-Ns> B4{} ;
  for (auto i = N - (Nb - Ns) ; i < N ; ++i)
    for (auto j = N - (Nb - Ns) ; j < N ; ++j)
      B4[i - (N - (Nb - Ns))][j - (N - (Nb - Ns))] = A[i][j] ;
  B4 = invert(B4);
  for (auto i = N - (Nb - Ns) ; i < N ; ++i)
    for (auto j = N - (Nb - Ns) ; j < N ; ++j)
      J[i][j] = B4[i - (N - (Nb - Ns))][j - (N - (Nb - Ns))] ;

  return J ;
}

template <std::size_t N, typename number=double>
constexpr auto invert(const auto& AA)
{
  std::array<std::array<number,N>,N> A{};
  for (auto j = 0u ; j < N ; j++)
    {
      std::array<number,N> x{};
      x[j] = 1. ;
      const auto x1 = AA*x;
      for (auto i = 0u ; i < N ; i ++)
      	{
      	  A[i][j] = x1[i] ;
      	}
    }
  return invert(A);
}

template <std::size_t Nb, std::size_t Ns=0, std::size_t N, typename number=double>
constexpr auto block_jacobi(const auto& AA)
{
  std::array<std::array<number,N>,N> A{};
  for (auto j = 0u ; j < N ; j++)
    {
      std::array<number,N> x{};
      x[j] = 1. ;
      const auto x1 = AA*x;
      for (auto i = 0u ; i < N ; i ++)
      	{
      	  A[i][j] = x1[i] ;
      	}
    }
  if (Nb == N) return invert(A);
  
  std::array<std::array<number,N>,N> J{} ;
  
  if constexpr (Ns != 0)
  		 {
  		   std::array<std::array<number,Ns>,Ns> B0{} ;
  		   for (auto i = 0u ; i < Ns ; ++i)
  		     for (auto j = 0u ; j < Ns ; ++j)
  		       B0[i][j] = A[i][j] ;
  		   B0 = invert(B0);
  		   for (auto i = 0u ; i < Ns ; ++i)
  		     for (auto j = 0u ; j < Ns ; ++j)
  		       J[i][j] = B0[i][j] ;
  		 }

  std::array<std::array<number,Nb>,Nb> B1{} ;
  for (auto i = Ns ; i < Ns + Nb ; ++i)
    for (auto j = Ns ; j < Ns + Nb ; ++j)
      B1[i - Ns][j - Ns] = A[i][j] ;
  B1 = invert(B1);
  for (auto i = Ns ; i < Ns + Nb ; ++i)
    for (auto j = Ns ; j < Ns + Nb ; ++j)
      J[i][j] = B1[i - Ns][j - Ns] ;

  std::array<std::array<number,Nb>,Nb> B2{} ;
  for (auto i = Ns ; i < Ns + Nb ; ++i)
    for (auto j = Ns ; j < Ns + Nb ; ++j)
      B2[i - Ns][j - Ns] = A[Nb+i][Nb+j] ;
  B2 = invert(B2);
  
  int par = (N >= 128) ? 1 : 0 ;
  #ifdef PARALLEL
  #pragma omp parallel for if(par) default(none) shared(J,B2)
  #endif
  for (auto b = 1u ; b < (N / Nb - 2) ; ++b)
    for (auto i = Ns ; i < Ns + Nb ; ++i)
      for (auto j = Ns ; j < Ns + Nb ; ++j)
  	J[b*Nb+i][b*Nb+j] = B2[i - Ns][j - Ns] ;

  std::array<std::array<number,Nb>,Nb> B3{} ;
  for (auto i = Ns ; i < Ns + Nb ; ++i)
    for (auto j = Ns ; j < Ns + Nb ; ++j)
      B3[i - Ns][j - Ns] = A[(N / Nb - 2)*Nb+i][(N / Nb - 2)*Nb+j] ;
  B3 = invert(B3);
  for (auto i = Ns ; i < Ns + Nb ; ++i)
    for (auto j = Ns ; j < Ns + Nb ; ++j)
      J[(N / Nb - 2)*Nb+i][(N / Nb - 2)*Nb+j] = B3[i - Ns][j - Ns] ;

  std::array<std::array<number,Nb-Ns>,Nb-Ns> B4{} ;
  for (auto i = N - (Nb - Ns) ; i < N ; ++i)
    for (auto j = N - (Nb - Ns) ; j < N ; ++j)
      B4[i - (N - (Nb - Ns))][j - (N - (Nb - Ns))] = A[i][j] ;
  B4 = invert(B4);
  for (auto i = N - (Nb - Ns) ; i < N ; ++i)
    for (auto j = N - (Nb - Ns) ; j < N ; ++j)
      J[i][j] = B4[i - (N - (Nb - Ns))][j - (N - (Nb - Ns))] ;

  return J ;
}

#endif /* ARRAYTOOLS_H */
