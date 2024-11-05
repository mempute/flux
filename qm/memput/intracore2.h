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
			dotv->joAxisPre[dotv->njoPre++] = jo_pre_ax[i];//����Ʈ ���� �ε��� ����.
			if(dotv->joTopAxisPre > jo_pre_ax[i]) dotv->joTopAxisPre = jo_pre_ax[i];//�ֻ���(0�� �����Ҽ��� ����) ����Ʈ ���� ����.
			jo_sz_pre *= fxp->fshape[jo_pre_ax[i]];
		}
		for(i = 0;i < dotv->njoPre; i++) axid[i] = fxp->fshape[*(dotv->joAxisPre + i)];//����Ʈ ���� ���������.
		Matrixr::make_rank_sz(dotv->njoPre, axid, dotv->joRankPre);//����Ʈ ���� ��ũ ������ ���.

		dotv->nJointAxis = 1;
		dotv->njoSuf = 0;
		for(i = 0;jo_suf_ax[i] >= 0; i++) {
			dotv->joAxisSuf[dotv->njoSuf++] = jo_suf_ax[i];
			if(dotv->joTopAxisSuf > jo_suf_ax[i]) dotv->joTopAxisSuf = jo_suf_ax[i];
			dotv->nJointAxis *= fxs->fshape[jo_suf_ax[i]];//����Ʈ �� ���� ���.
		}
		for(i = 0;i < dotv->njoSuf; i++) axid[i] = fxs->fshape[*(dotv->joAxisSuf + i)];
		Matrixr::make_rank_sz(dotv->njoSuf, axid, dotv->joRankSuf);

		if(jo_sz_pre != dotv->nJointAxis) throwFault(-1, "dot inconsistant shape");
		//if(bw_get_ori == 0) fxp->nJoint = fxs->nJoint = dotv->nJointAxis;

		for(i = dotv->noutPre = 0;out_pre_ax[i] >= 0; i++) {//����Ʈ�� �ƴ� ��� ���� �ε��� ����. �ֻ���(0�� �����Ҽ��� ����) ����Ʈ
			//if(out_pre_ax[i] > dotv->joTopAxisPre)
				dotv->outAxisPre[dotv->noutPre++] = out_pre_ax[i];//�������� ���� ������ �����Ѵ�.
		}
		if(dotv->noutPre) {//pref��Ʈ������ �ֻ��� ����Ʈ �������� ������ OFF_DOTM_TO_ORIGIN���� �ֻ��� ���� ��ũ�� szRankPre��
						//�ε����� ����ϹǷ� �ֻ��� ����Ʈ �������� �Ʒ� �������� ����Ʈ�� �ƴ� ������ ������츸 ��� ���� �����Ѵ�.
			for(i = 0;i < dotv->noutPre; i++) axid[i] = fxp->fshape[*(dotv->outAxisPre + i)];//��� ���� ���������.
			Matrixr::make_rank_sz(dotv->noutPre, axid, dotv->outRankPre);//��� ���� ��ũ ������ ���.
		}
		for(i = dotv->noutSuf = 0;out_suf_ax[i] >= 0; i++) {
			//if(out_suf_ax[i] > dotv->joTopAxisSuf)
				dotv->outAxisSuf[dotv->noutSuf++] = out_suf_ax[i];
		}
		if(dotv->noutSuf) {//suff��Ʈ������ �ֻ��� ����Ʈ �������� ������ OFF_DOTM_TO_ORIGIN���� �ֻ��� ���� ��ũ�� szRankSuf�� �ε�����
						//����ϹǷ� �ֻ��� ����Ʈ �������� �Ʒ� �������� ����Ʈ�� �ƴ� ������ ������츸 ��� ���� �����Ѵ�.
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
		} else if(bw_get_ori == BWDIV_SUFF) {//suffix, ��� ������ ��� ���� ���󰡰� szShrinkSuf ������ ������ �����࿡ ������ ������ �ö󰣴�.
			dotv->njoRet = fwDotv.njoSuf;
			dotv->noutRet = fwDotv.noutSuf;
			memcpy(dotv->joAxisRet, fwDotv.joAxisSuf, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outAxisRet, fwDotv.outAxisSuf, dotv->noutRet * sizeof(intt));
			memcpy(dotv->joRankRet, fwDotv.joRankSuf, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outRankRet, fwDotv.outRankSuf, dotv->noutRet * sizeof(intt));
			dotv->ndimRet = fwDotv.ndimSuf;
			memcpy(dotv->szRankRet, fwDotv.szRankSuf, dotv->ndimRet * sizeof(intt));
		}
		dotv->shareUnit = PDG;//dot ���� �������� ȣȯ�������� ����.
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
			jpre_dims[i] = fxp->fshape[jo_pre_ax[i]];//���� ������ ��������� pre�� suf�� ���ƾ� �Ϲǰ� ��ǥ�� pre���� ����.
			dotv->joAxisPre[i] = jo_pre_ax[i];//����Ʈ ���� �ε��� ����.
			dotv->sprPreJo[i].rkdim = fxp->fshape[jo_pre_ax[i]];
			dotv->sprPreJo[i].rksz = FX_SIZE_RK2(fxp, jo_pre_ax[i]) < 0 ? 0 : (fxp->fdim - 1 == jo_pre_ax[i] ? 1 : FX_SIZE_RK(fxp, jo_pre_ax[i] + 1));
			dotv->sprPreJo[i].rktsz = dotv->sprPreJo[i].rksz * (dotv->sprPreJo[i].rkdim - 1);
			jo_sz_pre *= fxp->fshape[jo_pre_ax[i]];
		}
		dotv->njoPre = i;
		for(i = 0;i < dotv->njoPre; i++) axid[i] = fxp->fshape[*(dotv->joAxisPre + i)];//����Ʈ ���� ���������.
		Matrixr::make_rank_sz(dotv->njoPre, axid, dotv->joRankPre);//����Ʈ ���� ��ũ ������ ���.

		dotv->nJointAxis = 1;
		dotv->jdimEqual = 1;
		for(i = 0;jo_suf_ax[i] >= 0; i++) {
			jsuf_dims[i] = fxs->fshape[jo_suf_ax[i]];
			if(jpre_dims[i] != jsuf_dims[i]) dotv->jdimEqual = 0;
			dotv->joAxisSuf[i] = jo_suf_ax[i];
			dotv->sprSufJo[i].rkdim = fxs->fshape[jo_suf_ax[i]];
			dotv->sprSufJo[i].rksz = FX_SIZE_RK2(fxs, jo_suf_ax[i]) < 0 ? 0 : (fxs->fdim - 1 == jo_suf_ax[i] ? 1 : FX_SIZE_RK(fxs, jo_suf_ax[i] + 1));
			dotv->sprSufJo[i].rktsz = dotv->sprSufJo[i].rksz * (dotv->sprSufJo[i].rkdim - 1);
			dotv->nJointAxis *= fxs->fshape[jo_suf_ax[i]];//����Ʈ �� ���� ���.
		}
		dotv->njoSuf = i;
		for(i = 0;i < dotv->njoSuf; i++) axid[i] = fxs->fshape[*(dotv->joAxisSuf + i)];
		Matrixr::make_rank_sz(dotv->njoSuf, axid, dotv->joRankSuf);

		if(jo_sz_pre != dotv->nJointAxis) throwFault(-1, "dot inconsistant shape");
		dotv->bwGetOri = bw_get_ori;
		if(bw_get_ori) {
			dotv->bwMxp = (fxp == dotOut ? TENSOR(fxp->quantum)->mxGrad : TENSOR(fxp->quantum)->mxData);
			dotv->bwMxs = (fxs == dotOut ? TENSOR(fxs->quantum)->mxGrad : TENSOR(fxs->quantum)->mxData);
			if(fwDotv.jdimEqual == 0) dotv->intervOut = 1;//�������� ���� ������� ��ġ���� ������ �����Ŀ��� ��ǥ ��ȯ�ϰ� �����Ѵ�.
		}// else fxp->nJoint = fxs->nJoint = dotv->nJointAxis;
		if(bw_get_ori == BWDIV_PREF) {//preffix
			dotv->njoRet = fwDotv.njoPre;
			dotv->noutRet = fwDotv.noutPre;
			memcpy(dotv->joAxisRet, fwDotv.joAxisPre, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outAxisRet, fwDotv.outAxisPre, dotv->noutRet * sizeof(intt));
			memcpy(dotv->joRankRet, fwDotv.joRankPre, dotv->njoRet * sizeof(intt));
			dotv->ndimRet = fwDotv.ndimPre;
			memcpy(dotv->szRankRet, fwDotv.szRankPre, dotv->ndimRet * sizeof(intt));
		} else if(bw_get_ori == BWDIV_SUFF) {//suffix, ��� ������ ��� ���� ���󰡰� szShrinkSuf ������ ������ �����࿡ ������ ������ �ö󰣴�.
			dotv->njoRet = fwDotv.njoSuf;
			dotv->noutRet = fwDotv.noutSuf;
			memcpy(dotv->joAxisRet, fwDotv.joAxisSuf, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outAxisRet, fwDotv.outAxisSuf, dotv->noutRet * sizeof(intt));
			memcpy(dotv->joRankRet, fwDotv.joRankSuf, dotv->njoRet * sizeof(intt));
			dotv->ndimRet = fwDotv.ndimSuf;
			memcpy(dotv->szRankRet, fwDotv.szRankSuf, dotv->ndimRet * sizeof(intt));
		}
		for(i = j = 0;out_pre_ax[i] >= 0; i++) {//����Ʈ�� �ƴ� ��� ���� �ε��� ����. �ֻ���(0�� �����Ҽ��� ����) ����Ʈ
			dotv->outAxisPre[i] = out_pre_ax[i];//�������� ���� ������ �����Ѵ�.
			dotv->outAxis[j] = out_pre_ax[i];
			dotv->sprPreOut[i].rkdim = fxp->fshape[out_pre_ax[i]];
			dotv->sprPreOut[i].rksz = FX_SIZE_RK2(fxp, out_pre_ax[i]) < 0 ? 0 : (fxp->fdim - 1 == out_pre_ax[i] ? 1 : FX_SIZE_RK(fxp, out_pre_ax[i] + 1));
			dotv->sprPreOut[i].rktsz = dotv->sprPreOut[i].rksz * (dotv->sprPreOut[i].rkdim - 1);
		}
		dotv->noutPre = i;
		if(dotv->noutPre) {//pref��Ʈ������ �ֻ��� ����Ʈ �������� ������ OFF_DOTM_TO_ORIGIN���� �ֻ��� ���� ��ũ�� szRankPre��  
						//�ε����� ����ϹǷ� �ֻ��� ����Ʈ �������� �Ʒ� �������� ����Ʈ�� �ƴ� ������ ������츸 ��� ���� �����Ѵ�.
			for(i = 0;i < dotv->noutPre; i++) axid[i] = fxp->fshape[*(dotv->outAxisPre + i)];//��� ���� ���������.
			Matrixr::make_rank_sz(dotv->noutPre, axid, dotv->outRankPre);//��� ���� ��ũ ������ ���.
		}
		for(i = 0;out_suf_ax[i] >= 0; i++) {
			dotv->outAxisSuf[i] = out_suf_ax[i];
			dotv->outAxis[j] = out_suf_ax[i];
			dotv->sprSufOut[i].rkdim = fxs->fshape[out_suf_ax[i]];
			dotv->sprSufOut[i].rksz = FX_SIZE_RK2(fxs, out_suf_ax[i]) < 0 ? 0 : (fxs->fdim - 1 == out_suf_ax[i] ? 1 : FX_SIZE_RK(fxs, out_suf_ax[i] + 1));
			dotv->sprSufOut[i].rktsz = dotv->sprSufOut[i].rksz * (dotv->sprSufOut[i].rkdim - 1);
		}
		dotv->noutSuf = i;
		if(dotv->noutSuf) {//suff��Ʈ������ �ֻ��� ����Ʈ �������� ������ OFF_DOTM_TO_ORIGIN���� �ֻ��� ���� ��ũ�� szRankSuf�� �ε����� 
						//����ϹǷ� �ֻ��� ����Ʈ �������� �Ʒ� �������� ����Ʈ�� �ƴ� ������ ������츸 ��� ���� �����Ѵ�.
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
	//A^: a0�� A�� first, a0�� b0�� ���� ���� b0�� first�� �ϸ� b1�� �� ������ �Ǽ� [b0,b1]�� pre���ǰ� 
	//		b1�� C matrix�� c1�̹Ƿ� [c1, c0]�� suf�� �ȴ�.
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
		//ax_out - �����Ķ� pre�� suf�� axis(�ε���)�� �����Ǵ� ��� ��Ʈ���� axis, ax_do - ��� ��Ʈ������ �ε����� ���� axis
		joint_opposit = p_that(ax_first, ax_join);
		if(joint_opposit) {//�����Ķ� first�� ���εǾ����� 
			ax_out_pre = joint_opposit;//first�� ���� matrix�� ������(���ε�) ��� matirx�� ��� axis�� pre out�� ����.
			ax_jo_pre = p_that(joint_opposit, ax_opposit);//��� matrix�� ���� axis�� ȹ��.
			i_do = i_this(ax_jo_pre, ax_out);//���� axis�� ��¸�Ʈ������ pre�� suf�� ������ΰ� ��å�Ͽ� �� �ε����� ȹ��.
			ax_jo_suf = ax_do[i_do];//ȹ��� ������ �ε����� ��� ��Ʈ���� ���� ���� axis�� suf join���� ����.
			ax_out_suf = ax_do[!i_do];//�� �ݴ븦 suf out���� ����.
			fx_pre = fx_opposit;//first�� ���� ��Ʈ������ �̹� ������ ���� preffix�� �ȴ�.
			fx_suf = fx_out;//�����Ķ��� ��� ��Ʈ������ �̹� ������ ���� suffix�� �ȴ�.
			//printf("%d(%d) %d(%d) %d(%d) %d(%d)\n", fx_pre->fshape[*ax_out_pre], *ax_out_pre,
			//	fx_pre->fshape[*ax_jo_pre], *ax_jo_pre, fx_suf->fshape[*ax_jo_suf], *ax_jo_suf,
			//	fx_suf->fshape[*ax_out_suf], *ax_out_suf);
			return 0;
		} else {//�����Ķ� first�� ���ε��� �ʾ����� ��¸�Ʈ������ ��µ����Ƿ� 
			i_do = i_this(ax_first, ax_out);//first�� ��¸�Ʈ������ pre�� suf�� ������ΰ� ��å�Ͽ� �� �ε����� ȹ��.
			ax_out_pre = ax_do[i_do];//�Ͽ� ��¸�Ʈ�������� �� �ε����� �ش��ϴ� axis�� pre out�� �����ϰ� 
			ax_jo_pre = ax_do[!i_do];//��¸�Ʈ�������� �� �ݴ븦 pre join���� ����.
			i_opposit = i_this(ax_out[!i_do], ax_opposit);//��� ��Ʈ�������� ���εǴ� axis �ε��� ȸ���Ͽ� 
			ax_jo_suf = ax_opposit[i_opposit];//��� ��Ʈ������ ���� axis ����
			ax_out_suf = ax_opposit[!i_opposit];//�� �ݴ븦 ��� axis�� ����.
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
