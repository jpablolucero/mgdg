#ifndef MFOPERATOR_H
#define MFOPERATOR_H

constexpr auto mfid = [](const auto v) {return v;};
  
template <typename Op, typename Preop=decltype(mfid)>
class MFOperator
{
public:

  constexpr MFOperator(const Op && op_) :
    op{std::forward<const Op>(op_)},
    preop{std::forward<const Preop>(mfid)}
  {}

  constexpr MFOperator(const Op && op_,const Preop && preop_) :
    op{std::forward<const Op>(op_)},
    preop{std::forward<const Preop>(preop_)}
  {}

  constexpr MFOperator(const Op & op_,const Preop & preop_=mfid) :
    op{op_},
    preop{preop_}
  {}

  constexpr MFOperator(const MFOperator & mfoperator):
    op(mfoperator.op),
    preop(mfoperator.preop)
  {}
  
  const Op op ;
  const Preop preop ;

  template <typename Op2, typename Preop2>
  constexpr auto operator()(const MFOperator<Op2,Preop2> & mfop) const 
  {
    return MFOperator<const MFOperator<Op,Preop>,const MFOperator<Op2,Preop2> >
      (MFOperator(op,preop),mfop);
  }

  constexpr auto operator()(const auto & v) const 
  {
    return op*(preop*v);
  }
}; 

#endif /* MFOPERATOR_H */
