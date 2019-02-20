#ifndef PROJECTIONS_H
#define PROJECTIONS_H

template <std::size_t N,std::size_t p=1,typename number=double>
constexpr auto restrict_matrix(const double factor = 1.)
{
  std::array<std::array<number,N>,N/2> R{};
  for (auto k = 0u ; k < N/2 ; k += p + 1)
    {
      bool avg = false ;
      std::size_t j = k ;
      for (auto i = 2*k ; i < 2*k+p+1 ; ++i)
	{
	  if (avg)
	    {
	      R[j][i] = 0.5 * factor ;
	      R[j+1][i] = 0.5 * factor ;
	      j += 1 ;
	      avg = false ;
	    }
	  else
	    {
	      R[j][i] = 1. * factor ;
	      avg = true ;
	    }
	}
      if (avg) avg = false ; else { avg = true ; j -= 1 ; }
      for (auto i = 2*k+p+1 ; i < 2*k+2*(p+1) ; ++i)
	{
	  if (avg)
	    {
	      R[j][i] = 0.5 * factor ;
	      R[j+1][i] = 0.5 * factor ;
	      j += 1 ;
	      avg = false ;
	    }
	  else
	    {
	      R[j][i] = 1. * factor ;
	      avg = true ;
	    }
	}
    }
  return R;
}

template <std::size_t N,std::size_t p=1,typename number=double>
class Restrict
{
public:
  constexpr Restrict(const number factor_=1.):
    factor(factor_)
  {}

  constexpr Restrict(const Restrict& restrict):
    factor(restrict.factor)
  {}
  
  const number factor ;
  
  constexpr auto operator()(const auto &v) const
  {
    std::array<number,N/2> w{};
    int par = (N >= 128) ? 1 : 0 ;
    #ifdef PARALLEL
    #pragma omp parallel for default(none) shared(w,v)
    #endif
    for (auto k = 0u ; k < N/2 ; k += p + 1)
      {
	bool avg = false ;
	std::size_t j = k ;
	for (auto i = 2*k ; i < 2*k+p+1 ; ++i)
	  {
	    if (avg)
	      {
		w[j] += 0.5 * factor * v[i] ;
		w[j+1] += 0.5 * factor * v[i] ;
		j += 1 ;
		avg = false ;
	      }
	    else
	      {
		w[j] += 1. * factor * v[i];
		avg = true ;
	      }
	  }
	if (avg) avg = false ; else { avg = true ; j -= 1 ; }
	for (auto i = 2*k+p+1 ; i < 2*k+2*(p+1) ; ++i)
	  {
	    if (avg)
	      {
		w[j] += 0.5 * factor * v[i];
		w[j+1] += 0.5 * factor * v[i];
		j += 1 ;
		avg = false ;
	      }
	    else
	      {
		w[j] += 1. * factor * v[i];
		avg = true ;
	      }
	  }
      }
    return w;
  }
};

template <std::size_t N,std::size_t p=1,typename number=double>
constexpr auto prolongate_matrix(const double factor = 1.)
{
  
  std::array<std::array<number,N/2>,N> RT{};
  for (auto k = 0u ; k < N/2 ; k += p + 1)
    {
      bool avg = false ;
      std::size_t j = k ;
      for (auto i = 2*k ; i < 2*k+p+1 ; ++i)
	{
	  if (avg)
	    {
	      RT[i][j] = 0.5 * factor ;
	      RT[i][j+1] = 0.5 * factor ;
	      j += 1 ;
	      avg = false ;
	    }
	  else
	    {
	      RT[i][j] = 1. * factor ;
	      avg = true ;
	    }
	}
      if (avg) avg = false ; else { avg = true ; j -= 1 ; }
      for (auto i = 2*k+p+1 ; i < 2*k+2*(p+1) ; ++i)
	{
	  if (avg)
	    {
	      RT[i][j] = 0.5 * factor ;
	      RT[i][j+1] = 0.5 * factor ;
	      j += 1 ;
	      avg = false ;
	    }
	  else
	    {
	      RT[i][j] = 1. * factor ;
	      avg = true ;
	    }
	}
    }
  return RT;
}

template <std::size_t N,std::size_t p=1,typename number=double>
class Prolongate
{
public:

  constexpr Prolongate(const number factor_ = 1.):
    factor(factor_)
  {}

  constexpr Prolongate(const Prolongate& prolongate):
    factor(prolongate.factor)
  {}

  const number factor ;
  
  constexpr auto operator()(const auto &v) const
  {
    std::array<number,N> w{};
    int par = (N >= 128) ? 1 : 0 ;
    #ifdef PARALLEL
    #pragma omp parallel for default(none) shared(w,v)
    #endif
    for (auto k = 0u ; k < N/2 ; k += p + 1)
      {
	bool avg = false ;
	std::size_t j = k ;
	for (auto i = 2*k ; i < 2*k+p+1 ; ++i)
	  {
	    if (avg)
	      {
		w[i] += 0.5 * factor * v[j];
		w[i] += 0.5 * factor * v[j+1];
		j += 1 ;
		avg = false ;
	      }
	    else
	      {
		w[i] += 1. * factor * v[j];
		avg = true ;
	      }
	  }
	if (avg) avg = false ; else { avg = true ; j -= 1 ; }
	for (auto i = 2*k+p+1 ; i < 2*k+2*(p+1) ; ++i)
	  {
	    if (avg)
	      {
		w[i] += 0.5 * factor * v[j];
		w[i] += 0.5 * factor * v[j+1];
		j += 1 ;
		avg = false ;
	      }
	    else
	      {
		w[i] += 1. * factor * v[j];
		avg = true ;
	      }
	  }
      }
    return w;
  }
};

#endif /* PROJECTIONS_H */
