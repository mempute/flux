#pragma once

#include "intracore.h"

/*class ApDot : public Apply {//dot version 1
public:
	Flux *dotPrefix, *dotSuffix, *dotOut;
	sytet transOrd;
	DotVar fwDotv, bwPreDotv, bwSufDotv;
	intt jo_ax_pre[MX_DIM], out_ax_pre[MX_DIM], jo_ax_suf[MX_DIM], out_ax_suf[MX_DIM], preout_do[MX_DIM], sufout_do[MX_DIM];

	void devideAxis(Flux *fx, vector<intt> *axis_jo, intt jo[], intt out[])
	{
		vector<intt>::iterator iter, end;
		intt i = 0, j = 0, k = 0, iax;

		for(;i < fx->fdim; i++) {
			for(iter = axis_jo->begin(), end = axis_jo->end(), iax = 0;iter != end && i != *iter; iter++, iax++);
			if(iter == end) out[k++] = i;
			else {
				jo[iax] = i;
				j++;
			}
		}
		out[k] = -1;
		jo[j] = -1;
	}
	void setDotv(DotVar *dotv, Flux *fxp, Flux *fxs, intt out_pre_ax[], intt jo_pre_ax[], intt jo_suf_ax[], intt out_suf_ax[], sytet bw_get_ori)
	{
		intt jo_sz_pre, axid[MX_DIM], i, j;

		dotv->intervOut = 0;
		dotv->joTopAxisPre = dotv->joTopAxisSuf = MX_DIM;
		dotv->njoPre = 0;

		jo_sz_pre = 1;
		for(i = 0;jo_pre_ax[i] >= 0; i++) {
			dotv->joAxisPre[dotv->njoPre++] = jo_pre_ax[i];//조인트 차원 인덱스 설정.
			if(dotv->joTopAxisPre > jo_pre_ax[i]) dotv->joTopAxisPre = jo_pre_ax[i];//최상위(0에 근접할수록 상위) 초인트 차원 설정.
			jo_sz_pre *= fxp->fshape[jo_pre_ax[i]];
		}
		for(i = 0;i < dotv->njoPre; i++) axid[i] = fxp->fshape[*(dotv->joAxisPre + i)];//조인트 차원 디멘젼설정.
		Matrixr::make_rank_sz(dotv->njoPre, axid, dotv->joRankPre);//조인트 차원 랭크 사이즈 계산.

		dotv->nJointAxis = 1;
		dotv->njoSuf = 0;
		for(i = 0;jo_suf_ax[i] >= 0; i++) {
			dotv->joAxisSuf[dotv->njoSuf++] = jo_suf_ax[i];
			if(dotv->joTopAxisSuf > jo_suf_ax[i]) dotv->joTopAxisSuf = jo_suf_ax[i];
			dotv->nJointAxis *= fxs->fshape[jo_suf_ax[i]];//조인트 총 갯수 계산.
		}
		for(i = 0;i < dotv->njoSuf; i++) axid[i] = fxs->fshape[*(dotv->joAxisSuf + i)];
		Matrixr::make_rank_sz(dotv->njoSuf, axid, dotv->joRankSuf);

		if(jo_sz_pre != dotv->nJointAxis) throwFault(-1, "dot inconsistant shape");
		//if(bw_get_ori == 0) fxp->nJoint = fxs->nJoint = dotv->nJointAxis;

		for(i = dotv->noutPre = 0;out_pre_ax[i] >= 0; i++) {//조인트가 아닌 출력 차원 인덱스 설정. 최상위(0에 근접할수록 상위) 조인트
			//if(out_pre_ax[i] > dotv->joTopAxisPre)
				dotv->outAxisPre[dotv->noutPre++] = out_pre_ax[i];//차원보다 하위 차원만 설정한다.
		}
		if(dotv->noutPre) {//pref매트릭스는 최상위 조인트 차원보다 상위는 OFF_DOTM_TO_ORIGIN에서 최상위 범위 랭크인 szRankPre로
						//인덱스를 계산하므로 최상위 조인트 차원보다 아래 차원에서 조인트가 아닌 차원이 있을경우만 출력 차원 설정한다.
			for(i = 0;i < dotv->noutPre; i++) axid[i] = fxp->fshape[*(dotv->outAxisPre + i)];//출력 차원 디멘젼설정.
			Matrixr::make_rank_sz(dotv->noutPre, axid, dotv->outRankPre);//출력 차원 랭크 사이즈 계산.
		}
		for(i = dotv->noutSuf = 0;out_suf_ax[i] >= 0; i++) {
			//if(out_suf_ax[i] > dotv->joTopAxisSuf)
				dotv->outAxisSuf[dotv->noutSuf++] = out_suf_ax[i];
		}
		if(dotv->noutSuf) {//suff매트릭스는 최상위 조인트 차원보다 상위는 OFF_DOTM_TO_ORIGIN에서 최상위 범위 랭크인 szRankSuf로 인덱스를
						//계산하므로 최상위 조인트 차원보다 아래 차원에서 조인트가 아닌 차원이 있을경우만 출력 차원 설정한다.
			for(i = 0;i < dotv->noutSuf; i++) axid[i] = fxs->fshape[*(dotv->outAxisSuf + i)];
			Matrixr::make_rank_sz(dotv->noutSuf, axid, dotv->outRankSuf);
		}
		dotv->ndimPre = fxp->fdim;
		dotv->ndimSuf = fxs->fdim;
		memcpy(dotv->szRankPre, TENSOR(fxp->quantum)->mxData->mxranksz, dotv->ndimPre * sizeof(intt));
		memcpy(dotv->szRankSuf, TENSOR(fxs->quantum)->mxData->mxranksz, dotv->ndimSuf * sizeof(intt));

		dotv->bwGetOri = bw_get_ori;
		if(bw_get_ori == BWDIV_PREF) {//preffix
			dotv->njoRet = fwDotv.njoPre;
			dotv->noutRet = fwDotv.noutPre;
			memcpy(dotv->joAxisRet, fwDotv.joAxisPre, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outAxisRet, fwDotv.outAxisPre, dotv->noutRet * sizeof(intt));
			memcpy(dotv->joRankRet, fwDotv.joRankPre, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outRankRet, fwDotv.outRankPre, dotv->noutRet * sizeof(intt));
			dotv->ndimRet = fwDotv.ndimPre;
			memcpy(dotv->szRankRet, fwDotv.szRankPre, dotv->ndimRet * sizeof(intt));
		} else if(bw_get_ori == BWDIV_SUFF) {//suffix, 출력 순서가 출력 축을 따라가고 szShrinkSuf 갯수를 넘으면 조인축에 단위가 순차로 올라간다.
			dotv->njoRet = fwDotv.njoSuf;
			dotv->noutRet = fwDotv.noutSuf;
			memcpy(dotv->joAxisRet, fwDotv.joAxisSuf, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outAxisRet, fwDotv.outAxisSuf, dotv->noutRet * sizeof(intt));
			memcpy(dotv->joRankRet, fwDotv.joRankSuf, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outRankRet, fwDotv.outRankSuf, dotv->noutRet * sizeof(intt));
			dotv->ndimRet = fwDotv.ndimSuf;
			memcpy(dotv->szRankRet, fwDotv.szRankSuf, dotv->ndimRet * sizeof(intt));
		}
		dotv->shareUnit = PDG;//dot 상위 버전과의 호환성때문에 설정.
	}
	ApDot(Trace *tcr, Flux *fxp, Flux *fxs, vector<intt> *axis_p, vector<intt> *axis_s, sytet trans_order, intt &fxo_ndim, intt fxo_axid[]) : Apply(tcr)
	{

		intt axid[MX_DIM], i;

		dotPrefix = fxp;
		dotSuffix = fxs;
		transOrd = trans_order;
		apCode = APC_DOT;
		nfanIn = 2;
		nfanOut = 1;

		devideAxis(fxp, axis_p, jo_ax_pre, out_ax_pre);
		devideAxis(fxs, axis_s, jo_ax_suf, out_ax_suf);

		for(i = fxo_ndim = 0;out_ax_pre[i] >= 0; i++) {
			preout_do[i] = fxo_ndim;
			fxo_axid[fxo_ndim++] = fxp->fshape[out_ax_pre[i]];
		}
		preout_do[i] = -1;
		for(i = 0;out_ax_suf[i] >= 0; i++) {
			sufout_do[i] = fxo_ndim;
			fxo_axid[fxo_ndim++] = fxs->fshape[out_ax_suf[i]];
		}
		sufout_do[i] = -1;

		registPhoto(fxp, fxs);
		listingApin(fxp);
		listingApin(fxs);
	}
	void *forward(TContext *tcxt)
	{
		Flux *fxr;
		sytet iv = invariance(dotPrefix, dotSuffix, fxr);
		if(iv == 0) dotOut->resizing2(fxr);
		if(iv <= 0) {
			setDotv(&fwDotv, dotPrefix, dotSuffix, out_ax_pre, jo_ax_pre, jo_ax_suf, out_ax_suf, 0);//[fxp(batch seq, feat)][fxs(feat, lattent)]=>[fxo(batch seq, lattent)]
			setDotv(&bwPreDotv, dotOut, dotSuffix, preout_do, sufout_do, out_ax_suf, jo_ax_suf, BWDIV_PREF);//[fxo(batch seq, lattent)][fxs^(lattent, feat)]->[fxp(batch seq, feat)]
			setDotv(&bwSufDotv, dotPrefix, dotOut, jo_ax_pre, out_ax_pre, preout_do, sufout_do, BWDIV_SUFF);//[fxp^(feat, batch seq)][fxo(batch seq, lattent)]->[fxs(feat, lattent)]
		}
		//printf("!!!!! DOT FORW \n");
		TENSOR(dotPrefix->quantum)->mxData->mdot(tcxt, TENSOR(dotSuffix->quantum)->mxData, TENSOR(dotOut->quantum)->mxData,
			&fwDotv, 0, nullx);//[fxp(batch seq, feat)][fxs(feat, lattent)]=>[fxo(batch seq, lattent)]
		return dotOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		//printf("!!!!! DOT BACK \n");
		bwLock(dotPrefix, bw_mutmx);
		TENSOR(dotOut->quantum)->mxGrad->mdot(tcxt, TENSOR(dotSuffix->quantum)->mxData, TENSOR(dotPrefix->quantum)->mxGrad,
			&bwPreDotv, TOB, (void *)1);//[fxo(batch seq, lattent)][fxs^(lattent, feat)]->[fxp(batch seq, feat)]
		bwUnlock(dotPrefix);
		bwLock(dotSuffix, bw_mutmx);
		TENSOR(dotPrefix->quantum)->mxData->mdot(tcxt, TENSOR(dotOut->quantum)->mxGrad, TENSOR(dotSuffix->quantum)->mxGrad,
			&bwSufDotv, TOA, (void *)1);//[fxp^(feat, batch seq)][fxo(batch seq, lattent)]->[fxs(feat, lattent)]
		bwUnlock(dotSuffix);
		return lapInput;
	}
};*/

class ApDot2 : public Apply {
public:
	Flux *dotPrefix, *dotSuffix, *dotOut;
	sytet transOrd;
	intt *preFirstAxis, *sufFirstAxis;
	DotVar fwDotv, bwPreDotv, bwSufDotv;
	intt jo_ax_pre[MX_DIM], out_ax_pre[MX_DIM], jo_ax_suf[MX_DIM], out_ax_suf[MX_DIM], preout_do[MX_DIM], sufout_do[MX_DIM];

	intt *devideAxis(Flux *fx, vector<intt> *axis_jo, intt jo[], intt out[], bool &interv)
	{
		vector<intt>::iterator iter, end;
		intt i = 0, j = 0, k = 0, iax;
		intt *first = nullx;

		interv = 0;
		for(;i < fx->fdim; i++) {
			for(iter = axis_jo->begin(), end = axis_jo->end(), iax = 0;iter != end && i != *iter; iter++, iax++);
			if(iter == end) {
				if(k != 0 && (i - out[k - 1] > 1 || i - out[k - 1] < 1)) interv = 1;
				out[k++] = i;
				if(i == 0) first = out;
			} else {
				if(iax != 0 && (i - jo[iax - 1] > 1 || i - jo[iax - 1] < 1)) interv = 1;
				jo[iax] = i;
				j++;
				if(i == 0) first = jo;
			}
		}
		out[k] = -1;
		jo[j] = -1;
		return first;
	}
	intt mk_rank_out(intt rank[], intt *po, intt *so, Flux *pre_fx, Flux *suf_fx)
	{
		intt axid[MX_DIM], i, n = 0;

		for(i = 0;*(po + i) >= 0; i++, n++) axid[n] = pre_fx->fshape[*(po + i)];
		for(i = 0;*(so + i) >= 0; i++, n++) axid[n] = suf_fx->fshape[*(so + i)];

		Matrixr::make_rank_sz(n, axid, rank);
		return n;
	}
	void setDotv(DotVar *dotv, Flux *fxp, Flux *fxs, intt out_pre_ax[], intt jo_pre_ax[], intt jo_suf_ax[], intt out_suf_ax[], sytet bw_get_ori)
	{
		intt jo_sz_pre, axid[MX_DIM], i, j, jpre_dims[MX_DIM], jsuf_dims[MX_DIM];

		dotv->useCublas = 0;
		jo_sz_pre = 1;
		for(i = 0;jo_pre_ax[i] >= 0; i++) {
			jpre_dims[i] = fxp->fshape[jo_pre_ax[i]];//조인 차원의 디멘젼들을 pre와 suf가 같아야 하므고 대표로 pre에서 설정.
			dotv->joAxisPre[i] = jo_pre_ax[i];//조인트 차원 인덱스 설정.
			dotv->sprPreJo[i].rkdim = fxp->fshape[jo_pre_ax[i]];
			dotv->sprPreJo[i].rksz = FX_SIZE_RK2(fxp, jo_pre_ax[i]) < 0 ? 0 : (fxp->fdim - 1 == jo_pre_ax[i] ? 1 : FX_SIZE_RK(fxp, jo_pre_ax[i] + 1));
			dotv->sprPreJo[i].rktsz = dotv->sprPreJo[i].rksz * (dotv->sprPreJo[i].rkdim - 1);
			jo_sz_pre *= fxp->fshape[jo_pre_ax[i]];
		}
		dotv->njoPre = i;
		for(i = 0;i < dotv->njoPre; i++) axid[i] = fxp->fshape[*(dotv->joAxisPre + i)];//조인트 차원 디멘젼설정.
		Matrixr::make_rank_sz(dotv->njoPre, axid, dotv->joRankPre);//조인트 차원 랭크 사이즈 계산.

		dotv->nJointAxis = 1;
		dotv->jdimEqual = 1;
		for(i = 0;jo_suf_ax[i] >= 0; i++) {
			jsuf_dims[i] = fxs->fshape[jo_suf_ax[i]];
			if(jpre_dims[i] != jsuf_dims[i]) dotv->jdimEqual = 0;
			dotv->joAxisSuf[i] = jo_suf_ax[i];
			dotv->sprSufJo[i].rkdim = fxs->fshape[jo_suf_ax[i]];
			dotv->sprSufJo[i].rksz = FX_SIZE_RK2(fxs, jo_suf_ax[i]) < 0 ? 0 : (fxs->fdim - 1 == jo_suf_ax[i] ? 1 : FX_SIZE_RK(fxs, jo_suf_ax[i] + 1));
			dotv->sprSufJo[i].rktsz = dotv->sprSufJo[i].rksz * (dotv->sprSufJo[i].rkdim - 1);
			dotv->nJointAxis *= fxs->fshape[jo_suf_ax[i]];//조인트 총 갯수 계산.
		}
		dotv->njoSuf = i;
		for(i = 0;i < dotv->njoSuf; i++) axid[i] = fxs->fshape[*(dotv->joAxisSuf + i)];
		Matrixr::make_rank_sz(dotv->njoSuf, axid, dotv->joRankSuf);

		if(jo_sz_pre != dotv->nJointAxis) throwFault(-1, "dot inconsistant shape");
		dotv->bwGetOri = bw_get_ori;
		if(bw_get_ori) {
			dotv->bwMxp = (fxp == dotOut ? TENSOR(fxp->quantum)->mxGrad : TENSOR(fxp->quantum)->mxData);
			dotv->bwMxs = (fxs == dotOut ? TENSOR(fxs->quantum)->mxGrad : TENSOR(fxs->quantum)->mxData);
			if(fwDotv.jdimEqual == 0) dotv->intervOut = 1;//순전파의 조인 디맨젼이 일치하지 않으면 역전파에서 좌표 변환하게 설정한다.
		}// else fxp->nJoint = fxs->nJoint = dotv->nJointAxis;
		if(bw_get_ori == BWDIV_PREF) {//preffix
			dotv->njoRet = fwDotv.njoPre;
			dotv->noutRet = fwDotv.noutPre;
			memcpy(dotv->joAxisRet, fwDotv.joAxisPre, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outAxisRet, fwDotv.outAxisPre, dotv->noutRet * sizeof(intt));
			memcpy(dotv->joRankRet, fwDotv.joRankPre, dotv->njoRet * sizeof(intt));
			dotv->ndimRet = fwDotv.ndimPre;
			memcpy(dotv->szRankRet, fwDotv.szRankPre, dotv->ndimRet * sizeof(intt));
		} else if(bw_get_ori == BWDIV_SUFF) {//suffix, 출력 순서가 출력 축을 따라가고 szShrinkSuf 갯수를 넘으면 조인축에 단위가 순차로 올라간다.
			dotv->njoRet = fwDotv.njoSuf;
			dotv->noutRet = fwDotv.noutSuf;
			memcpy(dotv->joAxisRet, fwDotv.joAxisSuf, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outAxisRet, fwDotv.outAxisSuf, dotv->noutRet * sizeof(intt));
			memcpy(dotv->joRankRet, fwDotv.joRankSuf, dotv->njoRet * sizeof(intt));
			dotv->ndimRet = fwDotv.ndimSuf;
			memcpy(dotv->szRankRet, fwDotv.szRankSuf, dotv->ndimRet * sizeof(intt));
		}
		for(i = j = 0;out_pre_ax[i] >= 0; i++) {//조인트가 아닌 출력 차원 인덱스 설정. 최상위(0에 근접할수록 상위) 조인트
			dotv->outAxisPre[i] = out_pre_ax[i];//차원보다 하위 차원만 설정한다.
			dotv->outAxis[j] = out_pre_ax[i];
			dotv->sprPreOut[i].rkdim = fxp->fshape[out_pre_ax[i]];
			dotv->sprPreOut[i].rksz = FX_SIZE_RK2(fxp, out_pre_ax[i]) < 0 ? 0 : (fxp->fdim - 1 == out_pre_ax[i] ? 1 : FX_SIZE_RK(fxp, out_pre_ax[i] + 1));
			dotv->sprPreOut[i].rktsz = dotv->sprPreOut[i].rksz * (dotv->sprPreOut[i].rkdim - 1);
		}
		dotv->noutPre = i;
		if(dotv->noutPre) {//pref매트릭스는 최상위 조인트 차원보다 상위는 OFF_DOTM_TO_ORIGIN에서 최상위 범위 랭크인 szRankPre로  
						//인덱스를 계산하므로 최상위 조인트 차원보다 아래 차원에서 조인트가 아닌 차원이 있을경우만 출력 차원 설정한다.
			for(i = 0;i < dotv->noutPre; i++) axid[i] = fxp->fshape[*(dotv->outAxisPre + i)];//출력 차원 디멘젼설정.
			Matrixr::make_rank_sz(dotv->noutPre, axid, dotv->outRankPre);//출력 차원 랭크 사이즈 계산.
		}
		for(i = 0;out_suf_ax[i] >= 0; i++) {
			dotv->outAxisSuf[i] = out_suf_ax[i];
			dotv->outAxis[j] = out_suf_ax[i];
			dotv->sprSufOut[i].rkdim = fxs->fshape[out_suf_ax[i]];
			dotv->sprSufOut[i].rksz = FX_SIZE_RK2(fxs, out_suf_ax[i]) < 0 ? 0 : (fxs->fdim - 1 == out_suf_ax[i] ? 1 : FX_SIZE_RK(fxs, out_suf_ax[i] + 1));
			dotv->sprSufOut[i].rktsz = dotv->sprSufOut[i].rksz * (dotv->sprSufOut[i].rkdim - 1);
		}
		dotv->noutSuf = i;
		if(dotv->noutSuf) {//suff매트릭스는 최상위 조인트 차원보다 상위는 OFF_DOTM_TO_ORIGIN에서 최상위 범위 랭크인 szRankSuf로 인덱스를 
						//계산하므로 최상위 조인트 차원보다 아래 차원에서 조인트가 아닌 차원이 있을경우만 출력 차원 설정한다.
			for(i = 0;i < dotv->noutSuf; i++) axid[i] = fxs->fshape[*(dotv->outAxisSuf + i)];
			Matrixr::make_rank_sz(dotv->noutSuf, axid, dotv->outRankSuf);
		}
		dotv->noutRank = dotv->noutPre + dotv->noutSuf;
		mk_rank_out(dotv->outRank, out_pre_ax, out_suf_ax, fxp, fxs);

		dotv->ndimPre = fxp->fdim;
		dotv->ndimSuf = fxs->fdim;
		memcpy(dotv->szRankPre, TENSOR(fxp->quantum)->mxData->mxranksz, dotv->ndimPre * sizeof(intt));
		memcpy(dotv->szRankSuf, TENSOR(fxs->quantum)->mxData->mxranksz, dotv->ndimSuf * sizeof(intt));
		dotv->shareUnit = PDG;
	}
	ApDot2(Trace *tcr, Flux *fxp, Flux *fxs, vector<intt> *axis_p, vector<intt> *axis_s, sytet trans_order, intt &fxo_ndim, intt fxo_axid[]) : Apply(tcr)
	{

		intt i;

		dotPrefix = fxp;
		dotSuffix = fxs;
		transOrd = trans_order;
		apCode = APC_DOT;
		nfanIn = 2;
		nfanOut = 1;

		preFirstAxis = devideAxis(fxp, axis_p, jo_ax_pre, out_ax_pre, bwPreDotv.intervOut);
		sufFirstAxis = devideAxis(fxs, axis_s, jo_ax_suf, out_ax_suf, bwSufDotv.intervOut);

		for(i = fxo_ndim = 0;out_ax_pre[i] >= 0; i++) {
			preout_do[i] = fxo_ndim;
			fxo_axid[fxo_ndim++] = fxp->fshape[out_ax_pre[i]];
		}
		preout_do[i] = -1;
		for(i = 0;out_ax_suf[i] >= 0; i++) {
			sufout_do[i] = fxo_ndim;
			fxo_axid[fxo_ndim++] = fxs->fshape[out_ax_suf[i]];
		}
		sufout_do[i] = -1;

		registPhoto(fxp, fxs);
		listingApin(fxp);
		listingApin(fxs);
	}
	//A: [a0, a1] B : [b0, b1]	[3,4] [3,2]		A: [a0, a1, a2] B: [b0, b1, b2]
	//J : [a1, a0] [b0, b1]		[4,3] [3,2]		J: [a0, a1, a2] * [b1, b0, b2]
	//C : [a1, b1]				[4,2]			C: [a0, b2]
	//A^ : [b0, b1] * [c1, c0]	[3,2] [2,4]		A^: [c0, c1] * [b2, b1, b0]
	//B^ : [a0, a1] * [c0, c1]	[3,4] [4,2]		B^: [a1, a2, a0] * [c0, c1]
	//A^: a0가 A의 first, a0는 b0와 같고 따라서 b0를 first로 하면 b1이 그 후행이 되서 [b0,b1]이 pre가되고 
	//		b1은 C matrix의 c1이므로 [c1, c0]가 suf가 된다.
	intt *p_that(intt *axis, intt *axis_pair[])
	{
		if(axis_pair[0] == axis) return axis_pair[1];
		else if(axis_pair[1] == axis) return axis_pair[0];
		else return nullx;
	}
	intt i_this(intt *axis, intt *axis_pair[])
	{
		if(axis_pair[0] == axis) return 0;
		else if(axis_pair[1] == axis) return 1;
		else exit(1);
	}
	bool bw_check_dot(Flux *fx_out, Flux *fx_opposit, intt *ax_first, intt *ax_opposit[], intt *ax_join[], intt *ax_out[],
		intt *ax_do[], intt *&ax_out_pre, intt *&ax_jo_pre, intt *&ax_jo_suf, intt *&ax_out_suf, Flux *&fx_pre, Flux *&fx_suf)
	{
		intt *joint_opposit, i_do, i_opposit;
		//ax_out - 순전파때 pre와 suf의 axis(인덱스)로 구성되는 출력 매트릭의 axis, ax_do - 출력 매트릭스의 인덱스를 갖는 axis
		joint_opposit = p_that(ax_first, ax_join);
		if(joint_opposit) {//순전파때 first로 조인되었으면 
			ax_out_pre = joint_opposit;//first가 속한 matrix와 곱해진(조인된) 상대 matirx의 출력 axis를 pre out로 설정.
			ax_jo_pre = p_that(joint_opposit, ax_opposit);//상대 matrix이 조인 axis을 획득.
			i_do = i_this(ax_jo_pre, ax_out);//조인 axis가 출력매트릭스의 pre와 suf중 어느쪽인가 검책하여 그 인덱스를 획득.
			ax_jo_suf = ax_do[i_do];//획득된 조인측 인덱스로 출력 매트릭스 상의 조인 axis를 suf join으로 설정.
			ax_out_suf = ax_do[!i_do];//그 반대를 suf out으로 설정.
			fx_pre = fx_opposit;//first의 상태 메트릭스가 이번 역전파 곲의 preffix가 된다.
			fx_suf = fx_out;//순전파때의 출력 매트릭스가 이번 역전파 곱의 suffix가 된다.
			//printf("%d(%d) %d(%d) %d(%d) %d(%d)\n", fx_pre->fshape[*ax_out_pre], *ax_out_pre,
			//	fx_pre->fshape[*ax_jo_pre], *ax_jo_pre, fx_suf->fshape[*ax_jo_suf], *ax_jo_suf,
			//	fx_suf->fshape[*ax_out_suf], *ax_out_suf);
			return 0;
		} else {//순전파때 first가 조인되지 않았으면 출력매트릭스에 출력됐으므로 
			i_do = i_this(ax_first, ax_out);//first가 출력매트릭스의 pre와 suf중 어느쪽인가 검책하여 그 인덱스를 획득.
			ax_out_pre = ax_do[i_do];//하여 출력매트릭스에서 그 인덱스에 해당하는 axis를 pre out로 설정하고 
			ax_jo_pre = ax_do[!i_do];//출력매트릭스에서 그 반대를 pre join으로 설정.
			i_opposit = i_this(ax_out[!i_do], ax_opposit);//상대 매트릭스에서 조인되는 axis 인덱스 회득하여 
			ax_jo_suf = ax_opposit[i_opposit];//상대 매트릭스의 조인 axis 설정
			ax_out_suf = ax_opposit[!i_opposit];//그 반대를 출력 axis로 설정.
			fx_pre = fx_out;
			fx_suf = fx_opposit;
			return 1;
		}
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		Flux *fxr;
		sytet iv = invariance(dotPrefix, dotSuffix, fxr);
		if(iv == 0) dotOut->resizing2(fxr, "ApDot2");
		if(iv <= 0) {
			setDotv(&fwDotv, dotPrefix, dotSuffix, out_ax_pre, jo_ax_pre, jo_ax_suf, out_ax_suf, 0);//[fxp(batch seq, feat)][fxs(feat, lattent)]=>[fxo(batch seq, lattent)]
			intt *ax_join[] = { jo_ax_pre, jo_ax_suf };
			intt *ax_out[] = { out_ax_pre, out_ax_suf };
			intt *ax_pre[] = { out_ax_pre, jo_ax_pre };
			intt *ax_suf[] = { jo_ax_suf, out_ax_suf };
			intt *ax_do[] = { preout_do, sufout_do };
			intt *po, *pj, *sj, *so;
			Flux *pre_fx, *suf_fx;
			//printf("%p %p %p %p\n", out_ax_pre, jo_ax_pre, jo_ax_suf, out_ax_suf);
			bwPreDotv.retFirst = bw_check_dot(dotOut, dotSuffix, preFirstAxis, ax_suf, ax_join, ax_out, ax_do, po, pj, sj, so, pre_fx, suf_fx);
			//printf("%p %p %p %p\n", po, pj, sj, so);
			setDotv(&bwPreDotv, pre_fx, suf_fx, po, pj, sj, so, BWDIV_PREF);//[fxo(batch seq, lattent)][fxs^(lattent, feat)]->[fxp(batch seq, feat)]
			bwSufDotv.retFirst = bw_check_dot(dotOut, dotPrefix, sufFirstAxis, ax_pre, ax_join, ax_out, ax_do, po, pj, sj, so, pre_fx, suf_fx);
			//printf("%p %p %p %p\n", po, pj, sj, so);
			setDotv(&bwSufDotv, pre_fx, suf_fx, po, pj, sj, so, BWDIV_SUFF);//[fxp^(feat, batch seq)][fxo(batch seq, lattent)]->[fxs(feat, lattent)]
		}
		//printf("!!!!! DOT FORW \n");
		
		TENSOR(dotPrefix->quantum)->mxData->mdot(tcxt, TENSOR(dotSuffix->quantum)->mxData, TENSOR(dotOut->quantum)->mxData,
			&fwDotv, 0, 0, nullx);//[fxp(batch seq, feat)][fxs(feat, lattent)]=>[fxo(batch seq, lattent)]
		multiArrangeUnlock();
		return dotOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		//printf("!!!!! DOT BACK \n");
		//arrangeLock(dotPrefix, bw_mutmx, lapInput, 0);
		((Matrixr *)bwPreDotv.bwMxp)->mdot(tcxt, (Matrixr *)bwPreDotv.bwMxs, TENSOR(dotPrefix->quantum)->mxGrad,
			&bwPreDotv, TOB, 0, (void *)1);//[fxo(batch seq, lattent)][fxs^(lattent, feat)]->[fxp(batch seq, feat)]
		//arrangeUnlock(dotPrefix, lapInput, 0);
		//arrangeLock(dotSuffix, bw_mutmx, lapInput->ptrRight, 0);
		((Matrixr *)bwSufDotv.bwMxp)->mdot(tcxt, (Matrixr *)bwSufDotv.bwMxs, TENSOR(dotSuffix->quantum)->mxGrad,
			&bwSufDotv, TOA, 0, (void *)1);//[fxp^(feat, batch seq)][fxo(batch seq, lattent)]->[fxs(feat, lattent)]
		//arrangeUnlock(dotSuffix, lapInput->ptrRight, 0);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
