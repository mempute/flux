#pragma once

#include "intracore.h"

enum class PaperCode
{
	FLUX,
	ARITH,
	DOT,
	MATMUL,
	SPLIT,
	UNSTACK,
	RESHAPE,
	COMBINATION,
	COMBINATION2,
	SQUEEZE,
	TRANSPOSE,
	SOFTMAX,
	SMCROSS_E,
	SQUARED_DIFF,
	SUM,
	MEAN,
	MEAN_S_E,
	ACTF,
	PRELU,
	SQRT,
	LOG,
	EMBEDING,
	ONEHOT,
	SLICE,
	ARGMAX,
	EQUAL,
	WRITEF,
	WRITET,
	COPYF,
	DSTW,
	ARANGE,
	FILL,
	RAND,
	MINIMIZE,
	ADMOPT,
	SGDOPT,
	CONCAT,
	STACK,
	BYPASS,
	PARTITION,
	ADJUST,
	LAYER_NORMAL,
	EXPOFILL,
	EXPANDE,
	SINPOS, 
	OVERWRITE,
	SWITCH_OUT,
	CLIP_VALUE,
	VMAX
};
#define TRCB(tcr) TRACER(tcr)->hbaper
#define BMAX_CSRGI	1024
class Baper {
public:
	bool playbap, ebaper, inner_flux;
	void *hbpalc;
	intt regicnt, bpalc, bpread, conscnt;
	Flux **fxregistry;
	void **csregistry;
	Trace *btrace;
	Baper *srcbap;
	intt savecnt;
	Baper(Trace *trc, sytet stepw)
	{
		btrace = trc;
		if(stepw < 0) {
			hbpalc = nullx;
			playbap = true;
		} else {
			hbpalc = rsc::mAllocator();
			playbap = false;
		}
		regicnt = conscnt = 0;
		bpalc = 0;
		bpread = -1;
		csregistry = (void **)malloc(sizeof(void *) * BMAX_CSRGI);
		fxregistry = nullx;
		ebaper = 0;
		inner_flux = 0;
	}
	~Baper()
	{
		if(playbap == false) rsc::rAllocator(hbpalc);
		free(csregistry);
		if(fxregistry) free(fxregistry);
	}
	void baperEnding(bool b)
	{
		ebaper = playbap = b;//paper build쓰기가 끝났으면(그라프 빌드가 끝났으면) 이후에 또 쓰기하는 것을 방지한다.
							//그라프 실행중에 flux의 fill과 같은 인터페이스 함수를 실행할 경우에 또 build paper쓰기 하지 않기위해
	}
	void instfxregi(Baper *src)
	{
		savecnt = src->regicnt;
		srcbap = src;
		fxregistry = (Flux **)malloc(sizeof(Flux *) * src->regicnt);
		src->baperwind();
	}
	intt *baperalc(intt rsz)
	{
		if(bpalc == bpread) return nullx;
		bpalc += rsz;
		return (intt *)rsc::ralloc2(hbpalc, rsz);
	}
	intt *baperalc_align(intt rsz)
	{
		if(bpalc == bpread) return nullx;
		bpalc += rsz;
		return (intt *)rsc::ralloc(hbpalc, rsz);
	}
	void baperwind()
	{
		bpread = bpalc;
		bpalc = 0;
		rsc::rrewind(hbpalc);
	}
	bool bobject(void *obj, intt &objid)
	{
		if(ebaper) return 1;

		objid = regicnt++;
		if(playbap) {
			*(fxregistry + objid) = (Flux *)obj;
			return true;
		}
		return false;
	}
	void bflux(Flux *fx, intt axid[], intt ndim, ubytet qtype, ubytet fxtype, vinitfp vfp, const bytet *name)
	{
		intt *p;

		if(ebaper) return;

		fx->bregid = regicnt++;
		if(playbap) {
			*(fxregistry + fx->bregid) = fx;
			return;
		}
		if(fxtype == memput::mp::apply || fxtype == memput::mp::const_apply || inner_flux) return;//apply에의해 생성되는 플럭스는 그라프 이주되는 측에서 생성할것이므로
												//플럭스 정보를 따로 또 전송하지 않게 한다.
		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::FLUX;
		p = baperalc(sizeof(intt));
		*p = ndim;
		for(intt i = 0;i < ndim; i++) {
			p = baperalc(sizeof(intt));
			*p = axid[i];
		}
		p = baperalc(sizeof(intt));
		*p = qtype;
		p = baperalc(sizeof(intt));
		*p = fxtype;
		p = baperalc(sizeof(intt));
		*p = (vfp ? vfp(nullx) : -1);
		p = baperalc(sizeof(intt));
		if(name) {
			intt len = strleng(name) + 1;
			*p = len;
			p = baperalc(len);
			strcpy((bytet *)p, name);
		} else *p = 0;
	}
	Flux *getflux(intt id)
	{
		return id < 0 ? nullx : *(fxregistry + id);
	}
	vinitfp getvfp(intt vid) //가중치 초기화, 변경은 마스터에서만 되고 슬레이븐 마스터 값이 복사되므로 필요없다. 나중에 삭제 검토.
	{
		switch(vid) {
		case T_XAVIER_INIT:
			return Initializer::xavier;
		case T_HE_INIT:
			return Initializer::he;
		case T_ONE_INIT:
			return Initializer::one;
		case T_ZERO_INIT:
			return Initializer::zero;
		default:
			return nullx;
		}
	}
	void *getcdata(intt id)
	{
		return *(csregistry + id);
	}
	void pflux(void)
	{
		intt len;
		intt axid[MX_DIM], ndim;
		ubytet qtype, fxtype;
		vinitfp vfp;
		bytet *name;

		ndim = *srcbap->baperalc(sizeof(intt));
		for(intt i = 0;i < ndim; i++) {
			axid[i] = *srcbap->baperalc(sizeof(intt));
		}
		qtype = *srcbap->baperalc(sizeof(intt));
		fxtype = *srcbap->baperalc(sizeof(intt));
		vfp = getvfp(*srcbap->baperalc(sizeof(intt)));
		len = *srcbap->baperalc(sizeof(intt));
		if(len) name = (bytet *)srcbap->baperalc(len);
		else name = nullx;

		new(btrace)Flux(btrace, axid, ndim, qtype, fxtype, vfp, (const bytet *)name);
	}
	void barith(Flux *fxp, Flux *fxs, ubytet qtype, void *sval, sytet arith_op)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::ARITH;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = (fxs ? fxs->bregid : -1);
		p = baperalc(sizeof(intt));
		*p = qtype;
		p = baperalc_align(sizeof(unit));
		*(unit *)p = (sval ? *(unit *)sval : 0);
		p = baperalc(sizeof(intt));
		*p = arith_op;
	}
	void parith(void)
	{
		Flux *fxp, *fxs;
		ubytet qtype;
		unit sval;
		sytet arith_op;

		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		fxs = getflux(*srcbap->baperalc(sizeof(intt)));

		qtype = *srcbap->baperalc(sizeof(intt));
		sval = *(unit *)srcbap->baperalc_align(sizeof(unit));
		arith_op = *srcbap->baperalc(sizeof(intt));
		
		fxp->arithmetic(fxs, qtype, fxs ? nullx : &sval, arith_op);
	}
	void bdot(Flux *fxp, Flux *fxs, vector<vector<intt> > axis, sytet trans_order)
	{
		intt *p;

		if(playbap) return;

		vector<intt> axis_p, axis_s;

		axis_p = axis.at(0);
		axis_s = axis.at(1);

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::DOT;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = fxs->bregid;
		p = baperalc(sizeof(intt));
		*p = (intt)axis_p.size();
		for(auto i : axis_p) {
			p = baperalc(sizeof(intt));
			*p = i;
		}
		p = baperalc(sizeof(intt));
		*p = (intt)axis_s.size();
		for(auto i : axis_s) {
			p = baperalc(sizeof(intt));
			*p = i;
		}
		p = baperalc(sizeof(intt));
		*p = trans_order;
	}
	void pdot(void)
	{
		Flux *fxp, *fxs;
		vector<vector<intt> > axis;
		vector<intt> axis_p, axis_s;
		sytet trans_order;
		intt i, n;

		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		fxs = getflux(*srcbap->baperalc(sizeof(intt)));

		n = *srcbap->baperalc(sizeof(intt));
		for(i = 0;i < n; i++) axis_p.push_back(*srcbap->baperalc(sizeof(intt)));
		n = *srcbap->baperalc(sizeof(intt));
		for(i = 0;i < n; i++) axis_s.push_back(*srcbap->baperalc(sizeof(intt)));
		axis.push_back(axis_p);
		axis.push_back(axis_s);
		trans_order = *srcbap->baperalc(sizeof(intt));
		
		fxp->dot(fxs, axis, trans_order);
	}
	void bmatmul(Flux *fxp, Flux *fxs, sytet trans_order)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::MATMUL;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = fxs->bregid;
		p = baperalc(sizeof(intt));
		*p = trans_order;
	}
	void pmatmul(void)
	{
		Flux *fxp, *fxs;
		sytet trans_order;

		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		fxs = getflux(*srcbap->baperalc(sizeof(intt)));
		trans_order = *srcbap->baperalc(sizeof(intt));

		fxp->matmul(fxs, trans_order);
	}
	void bsplit(Flux *fxp, intt nby, intt axis)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::SPLIT;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = nby;
		p = baperalc(sizeof(intt));
		*p = axis;
	}
	void psplit(void)
	{
		Flux *fxp;
		intt nby, axis;

		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		nby = *srcbap->baperalc(sizeof(intt));
		axis = *srcbap->baperalc(sizeof(intt));
		
		fxp->split(nby, axis);
	}
	void bscalar(void *sval, PaperCode pc)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)pc;
		p = baperalc(sizeof(unit));
		*(unit *)p = *(unit *)sval;
	}
	void pscalar(void *&sval)
	{
		sval = srcbap->baperalc(sizeof(unit));
	}
	void bfxvoid(Flux *fxp, PaperCode pc)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)pc;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
	}
	void pfxvoid(Flux *&fxp)
	{
		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
	}
	void bfxint(Flux *fxp, intt axis, PaperCode pc)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)pc;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = axis;
	}
	void pfxint(Flux *&fxp, intt &axis)
	{
		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		axis = *srcbap->baperalc(sizeof(intt));
	}
	void bfxintint(Flux *fxp, intt n, intt axis, PaperCode pc)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)pc;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = n;
		p = baperalc(sizeof(intt));
		*p = axis;
	}
	void pfxintint(Flux *&fxp, intt &n, intt &axis)
	{
		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		n = *srcbap->baperalc(sizeof(intt));
		axis = *srcbap->baperalc(sizeof(intt));
	}
	void bfxstr(Flux *fxp, const bytet *str, intt len, PaperCode pc)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)pc;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = len;
		p = baperalc(len);
		memcpy((bytet *)p, str, len);
	}
	void pfxstr(Flux *&fxp, bytet *&str, intt &len)
	{
		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		len = *srcbap->baperalc(sizeof(intt));
		str = (bytet *)srcbap->baperalc(len);
	}
	void bfxivect(Flux *fxp, vector<intt> axid, PaperCode pc)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)pc;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = (intt)axid.size();
		for(auto i : axid) {
			p = baperalc(sizeof(intt));
			*p = i;
		}
	}
	void pfxivect(Flux *&fxp, vector<intt> &axid)
	{
		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		intt i, len = *srcbap->baperalc(sizeof(intt));
		for(i = 0;i < len; i++) {
			axid.push_back(*srcbap->baperalc(sizeof(intt)));
		}
	}
	void bfxfx(Flux *fxp, Flux *fxs, PaperCode pc)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)pc;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = fxs->bregid;
	}
	void pfxfx(Flux *&fxp, Flux *&fxs)
	{
		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		fxs = getflux(*srcbap->baperalc(sizeof(intt)));
	}
	void bfxscalar(Flux *fxp, void *sval, PaperCode pc)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)pc;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc_align(sizeof(unit));
		*(unit *)p = *(unit *)sval;
	}
	void pfxscalar(Flux *&fxp, void *&sval)
	{
		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		sval = srcbap->baperalc_align(sizeof(unit));
	}
	void bfxssi(Flux *fxp, void *sval, void *sval2, intt i, PaperCode pc)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)pc;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc_align(sizeof(unit));
		*(unit *)p = *(unit *)sval;
		p = baperalc(sizeof(unit));
		*(unit *)p = *(unit *)sval2;
		p = baperalc(sizeof(intt));
		*p = i;
	}
	void pfxssi(Flux *&fxp, void *&sval, void *&sval2, intt &i)
	{
		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		sval = srcbap->baperalc_align(sizeof(unit));
		sval2 = srcbap->baperalc(sizeof(unit));
		i = *srcbap->baperalc(sizeof(intt));
	}
	void bvfxi(vector<Flux *> *fxl, intt iv, PaperCode pc)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)pc;
		p = baperalc(sizeof(intt));
		*p = (intt)fxl->size();
		for(auto pfx : *fxl) {
			p = baperalc(sizeof(intt));
			*p = pfx->bregid;
		}
		p = baperalc(sizeof(intt));
		*p = iv;
	}
	void pvfxi(vector<Flux *> *fxl, intt &iv)
	{
		intt n = *srcbap->baperalc(sizeof(intt));
		for(intt i = 0;i < n; i++) fxl->push_back(getflux(*srcbap->baperalc(sizeof(intt))));
		iv = *srcbap->baperalc(sizeof(intt));
	}
	void bunstack(Flux *fxp, intt axis)
	{
		bfxint(fxp, axis, PaperCode::UNSTACK);
	}
	void punstack(void)
	{
		Flux *fxp;
		intt axis;

		pfxint(fxp, axis);
		fxp->unstack(axis);
	}
	void breshape(Flux *fxp, vector<intt> axid)
	{
		bfxivect(fxp, axid, PaperCode::RESHAPE);
	}
	void preshape(void)
	{
		Flux *fxp;
		vector<intt> axid;

		pfxivect(fxp, axid);
		fxp->reshape(axid);
	}
	void bcombination(Flux *in, intx width, intx stride, doublex exc_contig_r, bool zero_pading, bool one_batch)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::COMBINATION;
		p = baperalc(sizeof(intt));
		*p = in->bregid;
		p = baperalc(sizeof(intt));
		*p = width;
		p = baperalc(sizeof(intt));
		*p = stride;
		p = baperalc(sizeof(intt));
		*p = zero_pading;
		p = baperalc(sizeof(intt));
		*p = one_batch;
		p = baperalc_align(sizeof(unit));
		*(doublet *)p = exc_contig_r;
	}
	void pcombination(void)
	{
		Flux *in;
		intt width, stride;
		doublet exc_contig_r;
		sytet zero_pading, one_batch;

		in = getflux(*srcbap->baperalc(sizeof(intt)));
		width = *srcbap->baperalc(sizeof(intt));
		stride = *srcbap->baperalc(sizeof(intt));
		zero_pading = *srcbap->baperalc(sizeof(intt));
		one_batch = *srcbap->baperalc(sizeof(intt));
		exc_contig_r = *(doublet *)srcbap->baperalc_align(sizeof(unit));

		btrace->tcr_combination(in, width, stride, exc_contig_r, zero_pading, one_batch);
	}
	void bcombination2(Flux *in, intx width, intx stride, doublex exc_contig_r, bool zero_pading, bool one_batch)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::COMBINATION2;
		p = baperalc(sizeof(intt));
		*p = in->bregid;
		p = baperalc(sizeof(intt));
		*p = width;
		p = baperalc(sizeof(intt));
		*p = stride;
		p = baperalc(sizeof(intt));
		*p = zero_pading;
		p = baperalc(sizeof(intt));
		*p = one_batch;
		p = baperalc_align(sizeof(unit));
		*(doublet *)p = exc_contig_r;
	}
	void pcombination2(void)
	{
		Flux *in;
		intt width, stride;
		doublet exc_contig_r;
		sytet zero_pading, one_batch;

		in = getflux(*srcbap->baperalc(sizeof(intt)));
		width = *srcbap->baperalc(sizeof(intt));
		stride = *srcbap->baperalc(sizeof(intt));
		zero_pading = *srcbap->baperalc(sizeof(intt));
		one_batch = *srcbap->baperalc(sizeof(intt));
		exc_contig_r = *(doublet *)srcbap->baperalc_align(sizeof(unit));

		btrace->tcr_combination2(in, width, stride, exc_contig_r, zero_pading, one_batch);
	}
	/*void bexpand_dims(Flux *fxp, intt axis) //depricate - reshape으로 수행되므로 따로 필요없다.
	{
		bfxint(fxp, axis, PaperCode::EXPDIMS); //depricate
	}
	void pexpand_dims(void) //depricate
	{
		Flux *fxp;
		intt axis;

		pfxint(fxp, axis);
		fxp->expand_dims(axis);
	}
	void bsqueeze(Flux *fxp, intt axis)
	{
		bfxint(fxp, axis, PaperCode::SQUEEZE); //depricate
	}
	void psqueeze(void)
	{
		Flux *fxp;
		intt axis;

		pfxint(fxp, axis);
		fxp->squeeze(axis);
	}*/
	void btranspose(Flux *fxp, vector<intt> axid)
	{
		bfxivect(fxp, axid, PaperCode::TRANSPOSE);
	}
	void ptranspose(void)
	{
		Flux *fxp;
		vector<intt> axid;

		pfxivect(fxp, axid);
		fxp->transpose(axid);
	}
	void bsoftmax(Flux *fxp)
	{
		bfxvoid(fxp, PaperCode::SOFTMAX);
	}
	void psoftmax(void)
	{
		Flux *fxp;

		pfxvoid(fxp);
		fxp->softmax();
	}
	void bsquaredDifference(Flux *fxp, Flux *fxs)
	{
		bfxfx(fxp, fxs, PaperCode::SQUARED_DIFF);
	}
	void psquaredDifference(void)
	{
		Flux *fxp, *fxs;
		pfxfx(fxp, fxs);
		fxp->squaredDifference(fxs);
	}
	void bsoftmaxCrossEntropy(Flux *fxp, Flux *fxt)
	{
		bfxfx(fxp, fxt, PaperCode::SMCROSS_E);
	}
	void psoftmaxCrossEntropy(void)
	{
		Flux *fxp, *fxt;
		pfxfx(fxp, fxt);
		fxp->softmaxCrossEntropy(fxt);
	}
	void bsum(Flux *fxp, intt batch_sum)
	{
		bfxint(fxp, batch_sum, PaperCode::SUM);
	}
	void psum(void)
	{
		Flux *fxp;
		intt batch_sum;

		pfxint(fxp, batch_sum);
		fxp->sum(batch_sum);
	}
	void bmean(Flux *fxp, intt batch_sum)
	{
		bfxint(fxp, batch_sum, PaperCode::MEAN);
	}
	void pmean(void)
	{
		Flux *fxp;
		intt batch_sum;

		pfxint(fxp, batch_sum);
		fxp->mean(batch_sum);
	}
	void bmeanSquareError(Flux *fxp, Flux *fxt, intt mean)
	{
		bfxfx(fxp, fxt, PaperCode::MEAN_S_E);
		intt *p = baperalc(sizeof(intt));
		*p = mean;
	}
	void pmeanSquareError(void)
	{
		Flux *fxp, *fxt;
		pfxfx(fxp, fxt);
		intt mean = *srcbap->baperalc(sizeof(intt));
		fxp->meanSquareError(fxt, mean);
	}
	void bactf(Flux *fxp, intt actf_op)
	{
		bfxint(fxp, actf_op, PaperCode::ACTF);
	}
	void pactf(void)
	{
		Flux *fxp;
		intt actf_op;

		pfxint(fxp, actf_op);
		fxp->actf(actf_op);
	}
	void bprelu(Flux *fxp, doublet dv)
	{
		bfxscalar(fxp, &dv, PaperCode::PRELU);
	}
	void pprelu(void)
	{
		Flux *fxp;
		void *p;

		pfxscalar(fxp, p);
		fxp->prelu(*(doublet *)p);
	}
	void bsqrt(Flux *fxp)
	{
		bfxvoid(fxp, PaperCode::SQRT);
	}
	void psqrt(void)
	{
		Flux *fxp;

		pfxvoid(fxp);
		fxp->sqrt();
	}
	void blog(Flux *fxp)
	{
		bfxvoid(fxp, PaperCode::LOG);
	}
	void plog(void)
	{
		Flux *fxp;

		pfxvoid(fxp);
		fxp->log();
	}
	void bembeding_lookup(Flux *fxp, Flux *fxi)
	{
		bfxfx(fxp, fxi, PaperCode::EMBEDING);
	}
	void pembeding_lookup(void)
	{
		Flux *fxp, *fxi;

		pfxfx(fxp, fxi);
		fxp->embedding_lookup(fxi);
	}
	void bone_hot(Flux *fxp, intt depth, doublet on_value, doublet off_value, intt axis, intt dtype)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::ONEHOT;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = depth;
		p = baperalc_align(sizeof(unit));
		*(unit *)p = *(unit *)&on_value;
		p = baperalc(sizeof(unit));
		*(unit *)p = *(unit *)&off_value;
		p = baperalc(sizeof(intt));
		*p = axis;
		p = baperalc(sizeof(intt));
		*p = dtype;
	}
	void pone_hot(void)
	{
		Flux *fxp;
		intt depth, axis, dtype;
		doublet on_value, off_value;

		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		depth = *srcbap->baperalc(sizeof(intt));
		on_value = *(doublet *)srcbap->baperalc_align(sizeof(unit));
		off_value = *(doublet *)srcbap->baperalc(sizeof(unit));
		axis = *srcbap->baperalc(sizeof(intt));
		dtype = *srcbap->baperalc(sizeof(intt));
		
		fxp->one_hot(depth, on_value, off_value, axis, dtype);
	}
	void bslice(Flux *fxp, vector<vector<intt>> axis)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::SLICE;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = (intt)axis.size();
		for(auto sub : axis) {
			p = baperalc(sizeof(intt));
			*p = (intt)sub.size();
			for(auto i : sub) {
				p = baperalc(sizeof(intt));
				*p = i;
			}
		}
	}
	void pslice(void)
	{
		Flux *fxp;
		vector<vector<intt>> axis;
		vector<intt> *sub;
		intt n, nsub, i, j;

		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		n = *srcbap->baperalc(sizeof(intt));
		for(i = 0;i < n; i++) {
			nsub = *srcbap->baperalc(sizeof(intt));
			sub = new vector<intt>;
			for(j = 0;j < nsub; j++) {
				sub->push_back(*srcbap->baperalc(sizeof(intt)));
			}
			axis.push_back(*sub);
			delete sub;
		}
		fxp->slice(axis);
	}
	void bargmax(Flux *fxp, intt axis)
	{
		bfxint(fxp, axis, PaperCode::ARGMAX);
	}
	void pargmax(void)
	{
		Flux *fxp;
		intt axis;

		pfxint(fxp, axis);
		fxp->argmax(axis);
	}
	void bvmax(Flux *fxp, intt axis)
	{
		bfxint(fxp, axis, PaperCode::VMAX);
	}
	void pvmax(void)
	{
		Flux *fxp;
		intt axis;

		pfxint(fxp, axis);
		fxp->vmax(axis);
	}
	void bequal(Flux *fxp, Flux *fxs, doublet cmpv, bool cscalr, bool eq)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::EQUAL;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = fxs ? fxs->bregid : -1;
		p = baperalc_align(sizeof(unit));
		*(unit *)p = *(unit *)&cmpv;
		p = baperalc(sizeof(intt));
		*p = cscalr;
		p = baperalc(sizeof(intt));
		*p = eq;
	}
	void pequal(void)
	{
		Flux *fxp, *fxs;
		doublet cmpv;
		bool cscalr, eq;

		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		fxs = getflux(*srcbap->baperalc(sizeof(intt)));
		cmpv = *(doublet *)srcbap->baperalc_align(sizeof(unit));
		cscalr = *srcbap->baperalc(sizeof(intt));
		eq = *srcbap->baperalc(sizeof(intt));
		
		fxp->equal(fxs, cmpv, cscalr, eq);
	}
	void bfeedf2(Flux *fxp, void *pdat, intt sz)
	{
		intt *p, id;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::WRITEF;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		id = conscnt++;
		p = baperalc(sizeof(intt));
		*p = id;
		p = baperalc(sizeof(intt));
		*p = sz;
		*(csregistry + id) = fxp->begin_p();//원격간 실행이면 이 슬롯의 상수 데이터들을 원격으로 전송.
	}
	void pfeedf2(void)
	{
		Flux *fxp;
		intt sz, cid;

		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		cid = *srcbap->baperalc(sizeof(intt));
		sz = *srcbap->baperalc(sizeof(intt));
		fxp->feedf(srcbap->getcdata(cid), sz);
	}
	void bfeedt(Flux *fxp, void *psrc, ubytet tsrc, intt sz)
	{
		intt *p, id;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::WRITET;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		id = conscnt++;
		p = baperalc(sizeof(intt));
		*p = id;
		p = baperalc(sizeof(intt));
		*p = tsrc;
		p = baperalc(sizeof(intt));
		*p = sz;
		*(csregistry + id) = fxp->begin_p();//원격간 실행이면 이 슬롯의 상수 데이터들을 원격으로 전송.
	}
	void pfeedt(void)
	{
		Flux *fxp;
		ubytet tsrc;
		intt sz, cid;

		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		cid = *srcbap->baperalc(sizeof(intt));
		tsrc = *srcbap->baperalc(sizeof(intt));
		sz = *srcbap->baperalc(sizeof(intt));
		fxp->feedt(srcbap->getcdata(cid), tsrc, sz);
	}
	void bfeedf(Flux *fxp, Flux *fxi)
	{
		bfxfx(fxp, fxi, PaperCode::COPYF);
	}
	void pfeedf(void)
	{
		Flux *fxp, *fxi;

		pfxfx(fxp, fxi);
		fxp->feedf(fxi);
	}
	void bdstrw(Flux *fxp, const bytet *dstr, intt len)
	{
		bfxstr(fxp, dstr, len, PaperCode::DSTW);
	}
	void pdstrw(void)
	{
		Flux *fxp;
		bytet *dstr;
		intt len;

		pfxstr(fxp, dstr, len);
		fxp->dstrw(dstr);
	}
	void barange(Flux *fxp, intt len)
	{
		bfxint(fxp, len, PaperCode::ARANGE);
	}
	void parange(void)
	{
		Flux *fxp;
		intt len;

		pfxint(fxp, len);
		fxp->arange(len);
	}
	void bfill(Flux *fxp, void *cv, ubytet tp, Flux *fxs)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::FILL;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc_align(sizeof(unit));
		*(unit *)p = *(unit *)cv;
		p = baperalc(sizeof(intt));
		*p = tp;
		p = baperalc(sizeof(intt));
		*p = (fxs ? fxs->bregid : -1);
	}
	void pfill(void)
	{
		Flux *fxp, *fxs;
		void *cv;
		ubytet tp;

		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		cv = srcbap->baperalc_align(sizeof(unit));
		tp = *srcbap->baperalc(sizeof(intt));
		fxs = getflux(*srcbap->baperalc(sizeof(intt)));
		
		if(tp == tdouble) fxp->fill(*(doublet *)cv, fxs);
		else if(tp == tfloat) fxp->fill(*(float *)cv, fxs);
		else if(tp == tlong) fxp->fill(*(longt *)cv, fxs);
	}
	void bsinpos(Flux *fxp, intt nseq, Flux *fxs)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::SINPOS;
		p = baperalc(sizeof(intt));
		*p = fxp->bregid;
		p = baperalc(sizeof(intt));
		*p = nseq;
		p = baperalc(sizeof(intt));
		*p = (fxs ? fxs->bregid : -1);
	}
	void psinpos(void)
	{
		Flux *fxp, *fxs;
		intt nseq;

		fxp = getflux(*srcbap->baperalc(sizeof(intt)));
		nseq = *srcbap->baperalc(sizeof(intt));
		fxs = getflux(*srcbap->baperalc(sizeof(intt)));

		fxp->sinpos(nseq);// , fxs);
	}
	void brandn(Flux *fxp, doublet m, doublet v)
	{
		bfxssi(fxp, &m, &v, RANDN_UNIV_OP, PaperCode::RAND);
	}
	void prand(void)
	{
		Flux *fxp;
		void *m, *v;
		intt rand_op;

		pfxssi(fxp, m, v, rand_op);
		if(rand_op == RANDN_UNIV_OP) fxp->randn(*(doublet *)m, *(doublet *)v);
		else if(rand_op == RANDU_UNIV_OP) fxp->randu(*(doublet *)m, *(doublet *)v);
	}
	void brandu(Flux *fxp, doublet m, doublet v)
	{
		bfxssi(fxp, &m, &v, RANDU_UNIV_OP, PaperCode::RAND);
	}
	void bminimize(Optimizer *opt, Flux *fxi)
	{
		intt *p;

		if(playbap) return;

		p = baperalc_align(sizeof(intt));
		*p = (intt)PaperCode::MINIMIZE;
		p = baperalc(sizeof(intt));
		*p = opt->bregid;
		p = baperalc(sizeof(intt));
		*p = fxi->bregid;
	}
	void pminimize(void)
	{
		Optimizer *opt;
		Flux *fxi;

		opt = (Optimizer *)getflux(*srcbap->baperalc(sizeof(intt)));
		fxi = getflux(*srcbap->baperalc(sizeof(intt)));

		opt->minimize(fxi);
	}
	void badam_optimizer(Optimizer *opt, doublet lr)
	{
		if(bobject(opt, opt->bregid)) return;

		bscalar(&lr, PaperCode::ADMOPT);
	}
	void padam_optimizer(void)
	{
		void *lr;

		pscalar(lr);
		adam_optimizer(btrace, *(doublet *)lr);
	}
	void bsgd_optimizer(Optimizer *opt, doublet lr)
	{
		if(bobject(opt, opt->bregid)) return;

		bscalar(&lr, PaperCode::SGDOPT);
	}
	void psgd_optimizer(void)
	{
		void *lr;

		pscalar(lr);
		gradient_descent_optimizer(btrace, *(doublet *)lr);
	}
	void bconcat(vector<Flux *> *fxl, intt axis)
	{
		bvfxi(fxl, axis, PaperCode::CONCAT);
	}
	void pconcat(void)
	{
		vector<Flux *> fxl;
		intt axis;

		pvfxi(&fxl, axis);
		concat(&fxl, axis);
	}
	void bstack(vector<Flux *> *fxl, intt axis)
	{
		bvfxi(fxl, axis, PaperCode::STACK);
	}
	void pstack(void)
	{
		vector<Flux *> fxl;
		intt axis;

		pvfxi(&fxl, axis);
		stack(&fxl, axis);
	}
	void bbypass(Flux *fxp)
	{
		bfxvoid(fxp, PaperCode::BYPASS);
	}
	void pbypass(void)
	{
		Flux *fxp;

		pfxvoid(fxp);
		fxp->bypass();
	}
	void bpartition(Flux *fxp)
	{
		bfxvoid(fxp, PaperCode::PARTITION);
	}
	void ppartition(void)
	{
		Flux *fxp;

		pfxvoid(fxp);
		fxp->partition();
	}
	void badjust(Flux *fxp, Flux *fxs)
	{
		bfxfx(fxp, fxs, PaperCode::ADJUST);
	}
	void padjust(void)
	{
		Flux *fxp, *fxs;
		pfxfx(fxp, fxs);
		fxp->adjust(fxs);
	}
	void blayer_normal(Flux *fxp)
	{
		bfxvoid(fxp, PaperCode::LAYER_NORMAL);
	}
	void player_normal(void)
	{
		Flux *fxp;

		pfxvoid(fxp);
		fxp->layer_normal();
	}
	void bexpofill(Flux *fxp, intt exp_c)
	{
		bfxint(fxp, exp_c, PaperCode::EXPOFILL);
	}
	void pexpofill(void)
	{
		Flux *fxp;
		intt exp_c;

		pfxint(fxp, exp_c);
		fxp->expofill(exp_c);
	}
	void bexpand_elen(Flux *fxp, intt n, intt axis)
	{
		bfxintint(fxp, n, axis, PaperCode::EXPANDE);
	}
	void pexpand_elen(void)
	{
		Flux *fxp;
		intt n, axis;

		pfxintint(fxp, n, axis);
		fxp->expand_elen(n, axis);
	}
	void boverwrite(Flux *fxp, Flux *fxs)
	{
		bfxfx(fxp, fxs, PaperCode::OVERWRITE);
	}
	void poverwrite(void)
	{
		Flux *fxp, *fxs;

		pfxfx(fxp, fxs);

		fxp->overwrite(fxs);
	}
	void bswitch_out(Flux *fxp, Flux *fxs)
	{
		bfxfx(fxp, fxs, PaperCode::SWITCH_OUT);
	}
	void pswitch_out(void)
	{
		Flux *fxp, *fxs;

		pfxfx(fxp, fxs);

		fxp->switchout(fxs);
	}
	void bclipval(Flux *fxp, doublet m, doublet v)
	{
		bfxssi(fxp, &m, &v, 0, PaperCode::CLIP_VALUE);
	}
	void pclipval(void)
	{
		Flux *fxp;
		void *m, *v;
		intt i;

		pfxssi(fxp, m, v, i);
		fxp->clipValue(*(doublet *)m, *(doublet *)v);
	}
	void transgraph(Baper *src)
	{
		instfxregi(src);

		intt *p = src->baperalc_align(sizeof(intt));

		while(p) {
			switch((PaperCode)*p) {
			case PaperCode::FLUX:
				pflux();
				break;
			case PaperCode::ARITH:
				parith();
				break;
			case PaperCode::DOT:
				pdot();
				break;
			case PaperCode::MATMUL:
				pmatmul();
				break;
			case PaperCode::SPLIT:
				psplit();
				break;
			case PaperCode::UNSTACK:
				punstack();
				break;
			case PaperCode::RESHAPE:
				preshape();
				break;
			case PaperCode::COMBINATION:
				pcombination();
				break;
			case PaperCode::COMBINATION2:
				pcombination2();
				break;
			case PaperCode::SQUEEZE://depricate
				//psqueeze();
				break;
			case PaperCode::TRANSPOSE:
				ptranspose();
				break;
			case PaperCode::SOFTMAX:
				psoftmax();
				break;
			case PaperCode::SMCROSS_E:
				psoftmaxCrossEntropy();
				break;
			case PaperCode::SUM:
				psum();
				break;
			case PaperCode::MEAN:
				pmean();
				break;
			case PaperCode::MEAN_S_E:
				pmeanSquareError();
				break;
			case PaperCode::ACTF:
				pactf();
				break;
			case PaperCode::PRELU:
				pprelu();
				break;
			case PaperCode::SQRT:
				psqrt();
				break;
			case PaperCode::LOG:
				plog();
				break;
			case PaperCode::EMBEDING:
				pembeding_lookup();
				break;
			case PaperCode::ONEHOT:
				pone_hot();
				break;
			case PaperCode::SLICE:
				pslice();
				break;
			case PaperCode::ARGMAX:
				pargmax();
				break;
			case PaperCode::EQUAL:
				pequal();
				break;
			case PaperCode::WRITEF:
				pfeedf2();
				break;
			case PaperCode::WRITET:
				pfeedt();
				break;
			case PaperCode::COPYF:
				pfeedf();
				break;
			case PaperCode::DSTW:
				pdstrw();
				break;
			case PaperCode::ARANGE:
				parange();
				break;
			case PaperCode::FILL:
				pfill();
				break;
			case PaperCode::RAND:
				prand();
				break;
			case PaperCode::MINIMIZE:
				pminimize();
				break;
			case PaperCode::ADMOPT:
				padam_optimizer();
				break;
			case PaperCode::SGDOPT:
				psgd_optimizer();
				break;
			case PaperCode::CONCAT:
				pconcat();
				break;
			case PaperCode::STACK:
				pstack();
				break;
			case PaperCode::BYPASS:
				pbypass();
				break;
			case PaperCode::PARTITION:
				ppartition();
				break;
			case PaperCode::ADJUST:
				padjust();
				break;
			case PaperCode::LAYER_NORMAL:
				player_normal();
				break;
			case PaperCode::EXPOFILL:
				pexpofill();
				break;
			case PaperCode::EXPANDE:
				pexpand_elen(); 
				break;
			case PaperCode::SINPOS:
				psinpos();
				break; 
			case PaperCode::OVERWRITE:
				poverwrite();
				break; 
			case PaperCode::SWITCH_OUT:
				pswitch_out();
				break;
			case PaperCode::CLIP_VALUE:
				pclipval();
				break;
			case PaperCode::VMAX:
				pvmax();
				break;
			}
			p = src->baperalc_align(sizeof(intt));
		}
		if(savecnt != regicnt) throwFault(-1, "grape migration fault\n");
	}
};
