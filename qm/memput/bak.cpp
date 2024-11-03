
template<typename T>
intt csplit_t(void *pcxt, T *m_split, intt pdim, intt sdim, intt split_size, intt rsplit_size, intt idx_origin,
	intt idx_width, intt axis, bool stacking, bool bw)
{
	ConcatVar *ccv = (ConcatVar *)pcxt;
	intt off = idx_origin * idx_width, poff;
	intt n = (split_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : split_size);
	intt idx[MX_DIM], pidx[MX_DIM], rsplit_off, *prank, *srank, i, j, si, soff;
	T *rsplit_mhost, *rsplit_mhosts;

	SET_LINK_VAR(ccv, pcxt);
	prank = P_LINK_VAR(intt, ccv, ccv->szRankPrimary);
	srank = P_LINK_VAR(intt, ccv, ccv->szRankSecondary);
	rsplit_mhosts = P_LINK_VAR(T, ccv, ccv->mptrHostSecondary);
	intt ssz = MRANK_SIZE(srank, 0);

	for(;off < n; off++) {
		si = off / ssz;
		soff = off % ssz;
		rsplit_mhost = *((T **)rsplit_mhosts + si);
		offset2idx(sdim, srank, soff, idx);
		if(stacking) {
			for(i = j = 0;i < pdim; i++) {
				if(i == axis) pidx[i] = si;
				else pidx[i] = idx[j++];
			}
			poff = idx2offset(pdim, prank, pidx);
		} else {
			idx[axis] += si * rsplit_size;
			poff = idx2offset(pdim, prank, idx);
		}
		if(bw) *(rsplit_mhost + soff) += *(m_split + poff);
		else *(rsplit_mhost + soff) = *(m_split + poff);
	}
	return n;
}
template<typename T>
__global__ void ksplit_f(void *pcxt, T *m_split, intt pdim, intt sdim, intt rsplit_size, intt idx_origin, intt idx_width, 
	intt axis, bool stacking, bool bw, intt n)
{
	intt off = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(off >= n) return;

	intt idx[MX_DIM], pidx[MX_DIM], rsplit_off, *prank, *srank, i, j, si, soff, poff;
	T *rsplit_mdev, *rsplit_mdevs;
	ConcatVar *ccv = (ConcatVar *)pcxt;

	SET_LINK_VAR(ccv, pcxt);
	prank = P_LINK_VAR(intt, ccv, ccv->szRankPrimary);
	srank = P_LINK_VAR(intt, ccv, ccv->szRankSecondary);
	rsplit_mdevs = P_LINK_VAR(T, ccv, ccv->mptrDevSecondary);
	intt ssz = MRANK_SIZE(srank, 0);
	//printf("%d %d %d %d\n", ccv->szRankPrimary, ccv->szRankSecondary, ccv->mptrDevSecondary, *prank);
	si = off / ssz;
	soff = off % ssz;
	rsplit_mdev = *((T **)rsplit_mdevs + si);
	doffset2idx(sdim, srank, soff, idx);
	if(stacking) {
		for(i = j = 0;i < pdim; i++) {
			if(i == axis) pidx[i] = si;
			else pidx[i] = idx[j++];
		}
		poff = didx2offset(pdim, prank, pidx);
	} else {
		idx[axis] += si * rsplit_size;
		poff = didx2offset(pdim, prank, idx);
	}
	if(bw) *(rsplit_mdev + soff) += *(m_split + poff);
	else *(rsplit_mdev + soff) = *(m_split + poff);
	//printf("[%d](%p) %d %d %f %f\n", off, rsplit_mdev, idx[axis], rsplit_off, *(m_split + off), *(rsplit_mdev + rsplit_off));
}

template<typename T>
intt gsplit_t(void *pcxt, T *m_split, intt pdim, intt sdim, intt split_size, intt rsplit_size, intt idx_origin, 
	intt idx_width, intt axis, bool stacking, bool bw)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (split_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : split_size);

	ksplit_f<T> << <grid, block >> > (pcxt, m_split, pdim, sdim, rsplit_size, idx_origin, idx_width, axis, stacking, bw, n);
	cudaThreadSynchronize();
	return n;
}
intt gsplit_f(void *pcxt, floatt *m_split, intt pdim, intt sdim, intt split_size, intt rsplit_size, intt idx_origin, 
	intt idx_width, intt axis, bool stacking, bool bw)
{
	return gsplit_t<floatt>(pcxt, m_split, pdim, sdim, split_size, rsplit_size, idx_origin, idx_width, axis, stacking, bw);
}
intt gsplit_f(void *pcxt, intt *m_split, intt pdim, intt sdim, intt split_size, intt rsplit_size, intt idx_origin, 
	intt idx_width, intt axis, bool stacking, bool bw)
{
	return gsplit_t<intt>(pcxt, m_split, pdim, sdim, split_size, rsplit_size, idx_origin, idx_width, axis, stacking, bw);
}
template<typename T>
intt cconcat_t(void *pcxt, T *m_rcat, intt pdim, intt sdim, intt rcat_size, intt idx_origin, intt idx_width, intt ncat,
	intt axis, bool stacking, bool bw)
{
	ConcatVar *ccv = (ConcatVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (rcat_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rcat_size);
	intt idx[MX_DIM], cat_off, i, j, k, *srank, *srank2, *pcat_axis_baseoff;
	T *pcat_mhost, *pcat_mhosts;

	SET_LINK_VAR(ccv, pcxt);
	srank = P_LINK_VAR(intt, ccv, ccv->szRankPrimary);
	srank2 = P_LINK_VAR(intt, ccv, ccv->szRankSecondary);
	pcat_axis_baseoff = P_LINK_VAR(intt, ccv, ccv->catAxisBaseoff);
	pcat_mhosts = P_LINK_VAR(T, ccv, ccv->mptrHostSecondary);

	for(;roff < n; roff++) {
		offset2idx(pdim, srank, roff, idx);
		if(stacking) {
			i = idx[axis];
			for(k = j = 0;k < pdim; k++) {
				if(k != axis) idx[j++] = idx[k];
			}
		} else {
			for(i = 0;i < ncat - 1 && *(pcat_axis_baseoff + i) < idx[axis]; i++);
			if(*(pcat_axis_baseoff + i) > idx[axis]) i--;//위 ncat -1체크 이유는 ncat로 체크하면 끝의 *pcat_axis_baseoff 체크가 메모리 폴트나므로
			idx[axis] -= *(pcat_axis_baseoff + i);
		}
		pcat_mhost = *((T **)pcat_mhosts + i);
		cat_off = idx2offset(sdim, srank2 + i * sdim, idx);//ㄱ.
		if(bw) *(m_rcat + roff) += *(pcat_mhost + cat_off);
		else *(m_rcat + roff) = *(pcat_mhost + cat_off);
		//printf("[%d](%p) %d %d %d %f %f\n", i, pcat_mhost, sdim, roff, cat_off, *(m_rcat + roff), *(pcat_mhost + cat_off));
	}
	return n - idx_origin * idx_width;
}
template<typename T>
__global__ void kconcat_f(void *pcxt, T *m_rcat, intt pdim, intt sdim, intt idx_origin, intt idx_width, intt ncat, 
	intt axis, bool stacking, bool bw, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;
	//intt is;

	if(roff >= n) return;

	intt idx[MX_DIM], cat_off, i = 0, j, k, *srank, *srank2, *pcat_axis_baseoff;
	T *pcat_mdev, *pcat_mdevs;
	ConcatVar *ccv = (ConcatVar *)pcxt;

	SET_LINK_VAR(ccv, pcxt);
	srank = P_LINK_VAR(intt, ccv, ccv->szRankPrimary);
	srank2 = P_LINK_VAR(intt, ccv, ccv->szRankSecondary);
	pcat_axis_baseoff = P_LINK_VAR(intt, ccv, ccv->catAxisBaseoff);
	pcat_mdevs = P_LINK_VAR(T, ccv, ccv->mptrDevSecondary);

	doffset2idx(pdim, srank, roff, idx);
	if(stacking) {
		i = idx[axis];
		for(k = j = 0;k < pdim; k++) {
			if(k != axis) idx[j++] = idx[k];
		}
	} else {
		for(;i < ncat - 1 && *(pcat_axis_baseoff + i) < idx[axis]; i++);
		if(*(pcat_axis_baseoff + i) > idx[axis]) i--;//위 ncat -1체크 이유는 ncat로 체크하면 끝의 *pcat_axis_baseoff 체크가 메모리 폴트나므로
		idx[axis] -= *(pcat_axis_baseoff + i);
	}
	pcat_mdev = *((T **)pcat_mdevs + i);
	//printf("[%d](%p)\n", i, pcat_mdev);
	//is = idx[axis];
	cat_off = didx2offset(sdim, srank2 + i * sdim, idx);//ㄱ.
	if(bw) *(m_rcat + roff) += *(pcat_mdev + cat_off);
	else *(m_rcat + roff) = *(pcat_mdev + cat_off);
	//printf("[%d](%p) %d %d %d %d %f %f\n", roff, pcat_mdev, i, is, idx[axis], cat_off, *(pcat_mdev + cat_off), *(m_rcat + roff));
}
template<typename T>
intt gconcat_t(void *pcxt, T *m_rcat, intt pdim, intt sdim, intt rcat_size, intt idx_origin, intt idx_width, intt ncat, 
	intt axis, bool stacking, bool bw)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (rcat_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rcat_size);

	kconcat_f<T> << <grid, block >> > (pcxt, m_rcat, pdim, sdim, idx_origin, idx_width, ncat, axis, stacking, bw, n);
	cudaThreadSynchronize();

	return n - idx_origin * idx_width;
}
intt gconcat_f(void *pcxt, floatt *m_rcat, intt pdim, intt sdim, intt rcat_size, intt idx_origin, intt idx_width, intt ncat,
	intt axis, bool stacking, bool bw)
{
	return gconcat_t<floatt>(pcxt, m_rcat, pdim, sdim, rcat_size, idx_origin, idx_width, ncat, axis, stacking, bw);
}
intt gconcat_f(void *pcxt, intt *m_rcat, intt pdim, intt sdim, intt rcat_size, intt idx_origin, intt idx_width, intt ncat, 
	intt axis, bool stacking, bool bw)
{
	return gconcat_t<intt>(pcxt, m_rcat, pdim, sdim, rcat_size, idx_origin, idx_width, ncat, axis, stacking, bw);
}
template<typename DT> class ConcatTrack : public RootTrack {
public:
	static Tracker *trkPool;
	Matrix<DT> *mtxPrimary, **mtxSecondary;
	intt nFactorCT, axisCT, sdimCT;
	bool ctStack, ctConcat, bwCatra;

	ConcatTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 1;
	}
	void mSetCatra(Matrix<DT> *primary, intt factor, intt axis, intt sdim, bool stacking, bool concat, void *psplit_mtx, bool bw)
	{
		mtxPrimary = primary;
		nFactorCT = factor;
		axisCT = axis;
		sdimCT = sdim;
		ctStack = stacking;
		ctConcat = concat;
		mtxSecondary = (Matrix<DT> **)psplit_mtx;
		bwCatra = bw;
	}
	void secondaryCopy(intt n, bool h2d)
	{
		intt off, si, base, wsz;
		intt ssz = MTX_SIZE((*mtxSecondary)), rest;

		for(off = idxOrigin * widthPer, rest = n - off;rest;) {
			si = off / ssz;
			base = off % ssz;
			wsz = ssz - base;
			if(rest < wsz) wsz = rest;
			if(h2d) (*(mtxSecondary + si))->copyHostToDevice(base, wsz);
			else (*(mtxSecondary + si))->copyDeviceToHost(base, wsz);
			off += wsz;
			rest -= wsz;
		}
	}
	void tracking(void)
	{
		intt n;
		chron_begin(lap, mtxPrimary->lapType == 1);
		if(execGpu) {//gpu로 실행할지를 결정하는 용도로만 사용.
			if(ctConcat) {
				n = gconcat_f(tcxtrTrk->mCxtDevice, mtxPrimary->mxmDevice, mtxPrimary->mxndim, sdimCT, MTX_SIZE(mtxPrimary), 
					idxOrigin, widthPer, nFactorCT, axisCT, ctStack, bwCatra);
				if(mtxPrimary->lapType == 1) printf("split[%d][%d] gpu: %lld\n", n, MTX_SIZE(mtxPrimary), chron_end(lap));
				mtxPrimary->copyDeviceToHost(idxOrigin * widthPer, n);
				rutra->puthGpu(execGpu, n);//kkk
			} else {
				n = gsplit_f(tcxtrTrk->mCxtDevice, mtxPrimary->mxmDevice, mtxPrimary->mxndim, sdimCT, MTX_SIZE(mtxPrimary),
					nFactorCT, idxOrigin, widthPer, axisCT, ctStack, bwCatra);
				if(mtxPrimary->lapType == 1) printf("split[%d][%d] gpu: %lld\n", n, MTX_SIZE(mtxPrimary), chron_end(lap));
				secondaryCopy(n, false);
				rutra->puthGpu(execGpu, n - idxOrigin * widthPer);
			}
		} else {
			if(ctConcat) {
				n = cconcat_t<DT>(tcxtrTrk->mCxtHost, mtxPrimary->mxmHost, mtxPrimary->mxndim, sdimCT, MTX_SIZE(mtxPrimary),
					idxOrigin, widthPer, nFactorCT, axisCT, ctStack, bwCatra);
				if(mtxPrimary->lapType == 1) printf("split[%d][%d] cpu: %lld\n", n, MTX_SIZE(mtxPrimary), chron_end(lap));
				mtxPrimary->copyHostToDevice(idxOrigin * widthPer, n);
				rutra->puthGpu(nullx, n);
			} else {
				n = csplit_t<DT>(tcxtrTrk->mCxtHost, mtxPrimary->mxmHost, mtxPrimary->mxndim, sdimCT, MTX_SIZE(mtxPrimary),
					nFactorCT, idxOrigin, widthPer, axisCT, ctStack, bwCatra);
				if(mtxPrimary->lapType == 1) printf("split[%d][%d] cpu: %lld\n", n, MTX_SIZE(mtxPrimary), chron_end(lap));
				secondaryCopy(n, true);
				rutra->puthGpu(nullx, n - idxOrigin * widthPer);
			}
		}
		srGate->srReturn();
	}
};
template<typename DT> Tracker *ConcatTrack<DT>::trkPool = nullx;

void msplit(TContext *tcxt, void *psplit_mtx, intt nsplit, intt axis, bool stacking, bool bw)
{
	Matrix<DT> *mtx = nullptr;
	SignalR *sr = rsc::srutra->srGet();
	ConcatTrack<DT> *ctrk;
	ConcatVar *ccv;
	void *dev_mptr, *host_mptr;
	intt i, *prank_size, n, width;
	unit lap = (lapType == 2 ? xucurrenttime() : 0);
	bool gpu = mxRutra->policyTrack(MTX_SIZE(this), 0, 0, PDC, width, n);

	tcxt->cxbegin();
	ccv = (ConcatVar *)tcxt->cxalloc(sizeof(ConcatVar), i);

	tcxt->cxalign();
	dev_mptr = tcxt->cxalloc(nsplit * sizeof(void *), ccv->mptrDevSecondary);
	host_mptr = tcxt->cxalloc(nsplit * sizeof(void *), ccv->mptrHostSecondary);

	for(i = 0;i < nsplit; i++) {
		mtx = *((Matrix<DT> **)psplit_mtx + i);
		*((DT **)dev_mptr + i) = mtx->mxmDevice;
		*((DT **)host_mptr + i) = mtx->mxmHost;
		//printf("%p\n", mtx->mxmDevice);
	}
	prank_size = tcxt->cxalloc(mxndim * sizeof(intt), ccv->szRankPrimary);
	memcpy(prank_size, mxranksz, mxndim * sizeof(intt));
	prank_size = tcxt->cxalloc(mtx->mxndim * sizeof(intt), ccv->szRankSecondary);
	memcpy(prank_size, mtx->mxranksz, mtx->mxndim * sizeof(intt));

	tcxt->syncCxt2Dev();

	for(i = 0;i < n; i++) {
		ctrk = (ConcatTrack<DT> *)rsc::srutra->trkGet<ConcatTrack<DT>>(tcxt, i, width, gpu);
		ctrk->mSetCatra(this, *(mxshape + axis) / nsplit, axis, mtx->mxndim, stacking, false, psplit_mtx, bw);
		ctrk->ontrack(sr, gpu);
	}
	sr->srWait();
	if(lapType == 2) printf("split[%d] lap: %lld\n", MTX_SIZE(this), xucurrenttime() - lap);
	sr->srPut<ConcatTrack<DT>>();
}
void mconcat(TContext *tcxt, void *pcat_mtx, intt ncat, intt axis, bool stacking, bool bw)
{
	Matrix<DT> *mtx = nullptr;
	SignalR *sr = rsc::srutra->srGet();
	intt i, *prank_size, *pcat_axis_baseoff, cat_axis_baseoff = 0, n, width;
	ConcatTrack<DT> *ctrk;
	ConcatVar *ccv;
	void *dev_mptr, *host_mptr;
	unit lap = (lapType == 2 ? xucurrenttime() : 0);
	bool gpu = mxRutra->policyTrack(MTX_SIZE(this), 0, 0, PDC, width, n);

	tcxt->cxbegin();
	ccv = (ConcatVar *)tcxt->cxalloc(sizeof(ConcatVar), i);

	prank_size = tcxt->cxalloc(mxndim * sizeof(intt), ccv->szRankPrimary);
	memcpy(prank_size, mxranksz, mxndim * sizeof(intt));

	mtx = *(Matrix<DT> **)pcat_mtx;
	prank_size = tcxt->cxalloc(ncat * mtx->mxndim * sizeof(intt), ccv->szRankSecondary);
	pcat_axis_baseoff = tcxt->cxalloc(ncat * sizeof(intt), ccv->catAxisBaseoff);
	tcxt->cxalign();
	dev_mptr = tcxt->cxalloc(ncat * sizeof(void *), ccv->mptrDevSecondary);
	host_mptr = tcxt->cxalloc(ncat * sizeof(void *), ccv->mptrHostSecondary);
	for(i = 0;i < ncat; i++) {
		mtx = *((Matrix<DT> **)pcat_mtx + i);
		*((DT **)dev_mptr + i) = mtx->mxmDevice;
		*((DT **)host_mptr + i) = mtx->mxmHost;
		//printf("%d %p\n", i, mtx->mxmDevice);
		memcpy(prank_size, mtx->mxranksz, mtx->mxndim * sizeof(intt));//이하 스템은 stack함수는 필요없으나
		prank_size += mtx->mxndim;//일관성 차원에서 또 kconcat_f의 ㄱ)을 따로 분리하지 않으므로 그냥 둔다.
		*pcat_axis_baseoff++ = cat_axis_baseoff;
		cat_axis_baseoff += *(mtx->mxshape + axis);
	}
	tcxt->syncCxt2Dev();
	for(i = 0;i < n; i++) {
		ctrk = (ConcatTrack<DT> *)rsc::srutra->trkGet<ConcatTrack<DT>>(tcxt, i, width, gpu);
		ctrk->mSetCatra(this, ncat, axis, mtx->mxndim, stacking, true, nullx, bw);
		ctrk->ontrack(sr, gpu);
	}
	sr->srWait();
	if(lapType == 2) printf("concat[%d] lap: %lld\n", MTX_SIZE(this), xucurrenttime() - lap);
	sr->srPut<ConcatTrack<DT>>();
}

typedef struct {
	intt ntrDims;
	intt trRankSrc[MX_DIM], trRankRet[MX_DIM], trTxid[MX_DIM];
} TransVar;
template<typename T>
intt ctrs_t(void *pcxt, T *m_strs, T *m_rtrs, intt r_size, intt idx_origin, intt idx_width, bool bw)
{
	TransVar *tsvar = (TransVar *)pcxt;
	intt roff = idx_origin * idx_width, coff;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt *srank = tsvar->trRankSrc, *rrank = tsvar->trRankRet, *txid = tsvar->trTxid;
	intt ndims = tsvar->ntrDims;
	intt soff, sidx[MX_DIM], ridx[MX_DIM], i;

	for(;roff < n; roff++) {
		offset2idx(ndims, rrank, roff, ridx);
		if(bw) {
			for(i = 0;i < ndims; i++) sidx[i] = ridx[*(txid + i)];
		} else {
			for(i = 0;i < ndims; i++) sidx[*(txid + i)] = ridx[i];
		}
		soff = idx2offset(ndims, srank, sidx);
		if(bw) *(m_rtrs + roff) += *(m_strs + soff);
		else *(m_rtrs + roff) = *(m_strs + soff);
		//offset2idx(ndims, srank, roff, sidx);
		//for(i = 0;i < ndims; i++) ridx[i] = sidx[*(txid + i)];
	}
	return n - idx_origin * idx_width;
}
class ApTranspose : public Apply {
public:
	Flux *tsOut, *tsIn;
	TransVar trsVar, bwTrsv;

	ApTranspose(Trace *tcr, Flux *ip, Flux *op, intt txid[]) : Apply(tcr)
	{
		apCode = APC_TRANSPOSE;
		tsIn = ip;
		tsOut = op;
		nfanIn = 1;
		nfanOut = 1;
		trsVar.ntrDims = op->fdim;
		memcpy(trsVar.trTxid, txid, sizeof(intt) * trsVar.ntrDims);
		bwTrsv.ntrDims = op->fdim;
		memcpy(bwTrsv.trTxid, txid, sizeof(intt) * bwTrsv.ntrDims);

		registPhoto(ip, nullx);
		listingApin(ip);
	}
	void *forward(TContext *tcxt)
	{
		sytet iv = invariance2(tsIn);
		if(iv == 0) {//첫번째 랭크가 변화됐는데
			for(intt i = 1;i < trsVar.ntrDims; i++) {//타겟에 다른 랭크로 전치되면 안된다.
				if(trsVar.trTxid[i] == 0) throwFault(-1, "first variable rank transposed error\n");
			}
			tsOut->resizing2(tsIn);
		}
		//printf("!!!!! TRANSPOSE FORW \n");
		if(iv <= 0) {
			memcpy(trsVar.trRankSrc, TENSOR(tsIn->quantum)->mxData->mxranksz, trsVar.ntrDims * sizeof(intt));
			memcpy(trsVar.trRankRet, TENSOR(tsOut->quantum)->mxData->mxranksz, trsVar.ntrDims * sizeof(intt));
			memcpy(bwTrsv.trRankSrc, TENSOR(tsOut->quantum)->mxData->mxranksz, bwTrsv.ntrDims * sizeof(intt));
			memcpy(bwTrsv.trRankRet, TENSOR(tsIn->quantum)->mxData->mxranksz, bwTrsv.ntrDims * sizeof(intt));
		}
		TENSOR(tsIn->quantum)->mxData->mtranspose(tcxt, TENSOR(tsOut->quantum)->mxData, &trsVar, 0);
		return tsOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		//printf("!!!!! TRANSPOSE BACK \n");
		bwLock(tsIn, bw_mutmx);
		TENSOR(tsOut->quantum)->mxGrad->mtranspose(tcxt, TENSOR(tsIn->quantum)->mxGrad, &bwTrsv, 1);
		bwUnlock(tsIn);
		return lapInput;
	}
};
template<typename T>
__global__ void ktrans_f(void *pcxt, T *m_strs, T *m_rtrs, intt idx_origin, intt idx_width, bool bw, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	TransVar *tsvar = (TransVar *)pcxt;
	intt *srank = tsvar->trRankSrc, *rrank = tsvar->trRankRet, *txid = tsvar->trTxid;
	intt ndims = tsvar->ntrDims;
	intt soff, sidx[MX_DIM], ridx[MX_DIM], i;

	doffset2idx(ndims, rrank, roff, ridx);
	if(bw) {
		for(i = 0;i < ndims; i++) sidx[i] = ridx[*(txid + i)];
	} else {
		for(i = 0;i < ndims; i++) sidx[*(txid + i)] = ridx[i];
	}
	soff = didx2offset(ndims, srank, sidx);
	if(bw) *(m_rtrs + roff) += *(m_strs + soff);
	else *(m_rtrs + roff) = *(m_strs + soff);
}

template<typename T>
intt cslice_t(void *pcxt, T *mpre, T *mret, intt r_size, intt idx_origin, intt idx_width, bool bw)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt *prank, *rrank, *slicer = ovar->idxOne;
	intt ndims = ovar->nrkPre;
	intt poff, pidx[MX_DIM], ridx[MX_DIM], i, j, k;

	if(bw) {//mret는 pre
		prank = ovar->rankOut;
		rrank = ovar->rankPre;
	} else {//mret는 out
		prank = ovar->rankPre;
		rrank = ovar->rankOut;
	}
	for(;roff < n; roff++) {
		offset2idx(ndims, rrank, roff, ridx);
		if(bw) {
			for(i = j = 0;i < ndims; i++, j += 3) {
				if(ridx[i] % slicer[j + 2]) {
					k = -1;
					break;
				}
				k = (ridx[i] / slicer[j + 2]);
				if(k < slicer[j] || k > slicer[j + 1]) {
					k = -1;
					break;
				}
				k -= slicer[j];
				pidx[i] = k;
			}
			if(k < 0) continue;//출력 포인트가 아닌 것은 스킵
			//printf("aaa %d\n", roff);
			poff = idx2offset(ndims, prank, pidx);
			*(mret + roff) += *(mpre + poff);
		} else {
			for(i = j = 0;i < ndims; i++, j += 3) {
				pidx[i] = slicer[j] + (ridx[i] * slicer[j + 2]);
			}
			poff = idx2offset(ndims, prank, pidx);
			*(mret + roff) = *(mpre + poff);
		}
		
	}
	return n - idx_origin * idx_width;
}
template<typename T>
__global__ void kslice_f(void *pcxt, T *mpre, T *mret, intt idx_origin, intt idx_width, bool bw, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	OneVar *ovar = (OneVar *)pcxt;
	intt *prank, *rrank, *slicer = ovar->idxOne;
	intt ndims = ovar->nrkPre;
	intt poff, pidx[MX_DIM], ridx[MX_DIM], i, j, k;

	if(bw) {//mret는 pre
		prank = ovar->rankOut;
		rrank = ovar->rankPre;
	} else {//mret는 out
		prank = ovar->rankPre;
		rrank = ovar->rankOut;
	}
	doffset2idx(ndims, rrank, roff, ridx);
	if(bw) {
		for(i = j = 0;i < ndims; i++, j += 3) {
			if(ridx[i] % slicer[j + 2]) {
				k = -1;
				break;
			}
			k = (ridx[i] / slicer[j + 2]);
			if(k < slicer[j] || k > slicer[j + 1]) {
				k = -1;
				break;
			}
			k -= slicer[j];
			pidx[i] = k;
		}
		if(k < 0) return;//출력 포인트가 아닌 것은 스킵
		poff = didx2offset(ndims, prank, pidx);
		*(mret + roff) += *(mpre + poff);
	} else {
		for(i = j = 0;i < ndims; i++, j += 3) {
			pidx[i] = slicer[j] + (ridx[i] * slicer[j + 2]);
		}
		poff = didx2offset(ndims, prank, pidx);
		*(mret + roff) = *(mpre + poff);
	}
}
class ApSlice : public Apply {
public:
	Flux *apPrefix, *apOut;
	intt codeSlice[MX_DIM * 3];
	sytet verginF;
	OneVar onev;

	void setSlicev(void)
	{
		onev.nrkPre = apPrefix->fdim;
		memcpy(onev.rankPre, TENSOR(apPrefix->quantum)->mxData->mxranksz, onev.nrkPre * sizeof(intt));
		onev.nrkOut = apOut->fdim;
		memcpy(onev.rankOut, TENSOR(apOut->quantum)->mxData->mxranksz, onev.nrkOut * sizeof(intt));
		apPrefix->boundSlice(apPrefix->fdim * 3, codeSlice, onev.idxOne, true);
	}
	ApSlice(Trace *tcr, Flux *fxp, Flux *fxo, intt code[]) : Apply(tcr)
	{
		apCode = APC_SLICE;
		apPrefix = fxp;
		apOut = fxo;
		nfanIn = 1;
		nfanOut = 1;
		memcpy(codeSlice, code, fxp->fdim * 3 * sizeof(intt));

		verginF = 1;

		registPhoto(apPrefix, nullx);
		listingApin(apPrefix);
	}
	void *forward(TContext *tcxt)
	{
		sytet iv = invariance2(apPrefix);
		if(iv == 0) {
			intt slicer_idx[3], nfirst;
			apPrefix->boundSlice(3, codeSlice, slicer_idx, false);
			nfirst = 1 + (slicer_idx[1] - slicer_idx[0]) / slicer_idx[2];
			apOut->resizing2(nfirst);//첫번째 랭크 사이즈를 계산하여 리사이징
		}
		if(iv <= 0 || verginF) {
			
			verginF = 0;
			setSlicev();
		}
		TENSOR(apPrefix->quantum)->mxData->mone(tcxt, nullx, TENSOR(apOut->quantum)->mxData, 2, &onev, AOP_SLICE, 0, PDC);
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *bw_mutmx)
	{
		bwLock(apPrefix, bw_mutmx);
		TENSOR(apOut->quantum)->mxGrad->mone(tcxt, nullx, TENSOR(apPrefix->quantum)->mxGrad, 2, &onev, AOP_SLICE, 1, PDC);
		bwUnlock(apPrefix);
		return lapInput;
	}
};
/*
template<typename T>
intt conehot_t(void *pcxt, T *mpre, T *mret, intt r_size, intt idx_origin, intt idx_width)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt poff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt *prank = ovar->rankPre, *rrank = ovar->rankOut, axis = ovar->idxOne[4];
	T onv = (T)*(doublet *)ovar->idxOne;
	intt pdim = ovar->nrkPre, rdim = ovar->nrkOut;
	intt roff, pidx[MX_DIM], ridx[MX_DIM], i, j, depth = ovar->idxOne[5];

	offset2idx(pdim, prank, poff, pidx);
	const intt tdim = ovar->dimPre[--pdim];
	goto LB0;
	for(;poff < n; poff++) {
		for(;;) {
			if(tdim == ++pidx[pdim]) {//tdim은 맨 하위에 위치하는 것의 디멘젼, it, tsz도 마찬가지)
				pidx[pdim] = 0;
				i = pdim;
LB1:			if(--i < 0) goto LB2;
				else {
					if(ovar->dimPre[i] == ++pidx[i]) {
						pidx[i] = 0;
						goto LB1;
					} else break;
				}
			} else break;
		}
LB0:;	if(*(mpre + poff) >= depth || *(mpre + poff) < 0) continue;
		for(i = j = 0;i < rdim; i++) {
			if(i == axis) ridx[i] = *(mpre + poff);
			else ridx[i] = pidx[j++];
		}
		roff = idx2offset(rdim, rrank, ridx);
		*(mret + roff) = onv;
	}
LB2:;
	return n - idx_origin * idx_width;
}*/
template<typename T>
intt conehot_t(void *pcxt, T *mpre, T *mret, intt r_size, intt idx_origin, intt idx_width)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt poff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt *prank = ovar->rankPre, *rrank = ovar->rankOut, axis = ovar->idxOne[4];
	T onv = (T)*(doublet *)ovar->idxOne, offv = (T)*(doublet *)&ovar->idxOne[2];
	intt pdim = ovar->nrkPre, rdim = ovar->nrkOut;
	intt roff, pidx[MX_DIM], ridx[MX_DIM], i, j, depth = ovar->idxOne[5];

	for(;poff < n; poff++) {
		if(*(mpre + poff) >= depth || *(mpre + poff) < 0) continue;
		offset2idx(pdim, prank, poff, pidx);
		for(i = j = 0;i < rdim; i++) {
			if(i == axis) ridx[i] = *(mpre + poff);
			else ridx[i] = pidx[j++];
		}
		roff = idx2offset(rdim, rrank, ridx);
		*(mret + roff) = onv;
	}
	return n - idx_origin * idx_width;
}
template<typename T>
__global__ void konehot_f(void *pcxt, T *mpre, T *mret, intt idx_origin, intt idx_width, intt n)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt poff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x, depth = ovar->idxOne[5];

	if(poff >= n || *(mpre + poff) >= depth || *(mpre + poff) < 0) return;
	
	intt *prank = ovar->rankPre, *rrank = ovar->rankOut, axis = ovar->idxOne[4];
	T onv = (T)*(doublet *)ovar->idxOne;
	intt pdim = ovar->nrkPre, rdim = ovar->nrkOut;
	intt roff, pidx[MX_DIM], ridx[MX_DIM], i, j;
	
	doffset2idx(pdim, prank, poff, pidx);
	for(i = j = 0;i < rdim; i++) {
		if(i == axis) ridx[i] = *(mpre + poff);
		else ridx[i] = pidx[j++];
	}
	roff = didx2offset(rdim, rrank, ridx);
	*(mret + roff) = onv;
}
class ApOneHot : public Apply {
public:
	Flux *apPrefix, *apOut;
	sytet verginF;
	intt axisoh, depthoh;
	doublet onvoh, offvoh;
	OneVar onev;

	void setohv(doublet on_value, doublet off_value, intt axis, intt depth)
	{
		onev.nrkPre = apPrefix->fdim;
		memcpy(onev.rankPre, TENSOR(apPrefix->quantum)->mxData->mxranksz, onev.nrkPre * sizeof(intt));
		memcpy(onev.dimPre, apPrefix->fshape, onev.nrkPre * sizeof(intt));
		onev.nrkOut = apOut->fdim;
		memcpy(onev.rankOut, TENSOR(apOut->quantum)->mxData->mxranksz, onev.nrkOut * sizeof(intt));
		*(doublet *)onev.idxOne = on_value;
		*(doublet *)&onev.idxOne[2] = off_value;
		onev.idxOne[4] = axis;
		onev.idxOne[5] = depth;
	}
	ApOneHot(Trace *tcr, Flux *fxp, Flux *fxo, doublet on_value, doublet off_value, intt axis, intt depth) : Apply(tcr)
	{
		apCode = APC_ONEHOT;
		apPrefix = fxp;
		apOut = fxo;
		nfanIn = 1;
		nfanOut = 1;
		onvoh = on_value;
		offvoh = off_value;
		axisoh = axis;
		depthoh = depth;
		verginF = 1;

		registPhoto(apPrefix, nullx);
		listingApin(apPrefix);
	}
	void *forward(TContext *tcxt)
	{
		sytet iv = invariance2(apPrefix);
		if(iv == 0) {
			if(axisoh == 0) throwFault(-1, "first variable rank encoding error\n");
			apOut->resizing2(apPrefix);
		}
		if(iv <= 0 || verginF) {
			verginF = 0;
			setohv(onvoh, offvoh, axisoh, depthoh);
		}
		apOut->fill(*(doublet *)&onev.idxOne[2]);//off value먼저 일괄 설정
		TENSOR(apPrefix->quantum)->mxData->mone(tcxt, nullx, TENSOR(apOut->quantum)->mxData, 0, &onev, AOP_ONEHOT, 0, PDC);
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *bw_mutmx)
	{
		return nullx;
	}
};
template<typename T>
intt cargmax_t(void *pcxt, T *mpre, T *mret, intt r_size, intt idx_origin, intt idx_width)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt *prank = ovar->rankPre, *rrank = ovar->rankOut, axis = ovar->idxOne[0];
	intt pdim = ovar->nrkPre, rdim = ovar->nrkOut;
	intt poff, pidx[MX_DIM], ridx[MX_DIM], i, j, naxis = ovar->dimPre[axis];
	T vmax;

	for(;roff < n; roff++) {
		offset2idx(rdim, rrank, roff, ridx);
		for(i = j = 0;i < pdim; i++) {
			if(i != axis) pidx[i] = ridx[j++];
		}
		vmax = 0;
		for(i = 0;i < naxis; i++) {
			pidx[axis] = i;
			poff = idx2offset(pdim, prank, pidx);
			if(vmax < *(mpre + poff)) {
				vmax = *(mpre + poff);
				*(mret + roff) = i;
			}
		}
	}
	return n - idx_origin * idx_width;
}
template<typename T>
__global__ void kargmax_f(void *pcxt, T *mpre, T *mret, intt idx_origin, intt idx_width, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	OneVar *ovar = (OneVar *)pcxt;
	intt *prank = ovar->rankPre, *rrank = ovar->rankOut, axis = ovar->idxOne[0];
	intt pdim = ovar->nrkPre, rdim = ovar->nrkOut;
	intt poff, pidx[MX_DIM], ridx[MX_DIM], i, j, naxis = ovar->dimPre[axis];
	T vmax;

	doffset2idx(rdim, rrank, roff, ridx);
	for(i = j = 0;i < pdim; i++) {
		if(i != axis) pidx[i] = ridx[j++];
	}
	vmax = 0;
	for(i = 0;i < naxis; i++) {
		pidx[axis] = i;
		poff = didx2offset(pdim, prank, pidx);
		if(vmax < *(mpre + poff)) {
			vmax = *(mpre + poff);
			*(mret + roff) = i;
		}
	}
}
Flux *Flux::argmax(intt axis)
{
	intt axid[MX_DIM], i, j;
	OneVar onev;

	TRCB(fxTcr)->bargmax(this, axis);

	for(i = j = 0;i < fdim; i++) {
		if(axis == i) continue;
		axid[j++] = fshape[i];
	}
	onev.idxOne[0] = axis;
	Flux *fxo = new(fxTcr)Flux(fxTcr, axid, j, qType, apply);
	ApOne *ap = new(fxTcr)ApOne(TRACER(fxTcr), this, nullx, fxo, &onev, true, 2, -1, AOP_ARGMAX, 0, axis, 1, PDC);

	backend_ps(this, nullx, fxo, ap);
	return fxo;
}

template<typename T>
intt cembedding_t(T *mpre, T *msuf, T *mret, intt r_size, intt idx_origin, intt idx_width, intt sz_embed, intt stp, intt bw)
{
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size), idx;

	if(bw) {//msuf - input, mret - lookup table, mpre - embeded, roff는 mpre 기준
		for(;roff < n; roff++) {
			int_val_type(idx, &msuf[roff / sz_embed], stp);
			mret[idx*sz_embed + roff % sz_embed] += mpre[roff];
			//printf("%d %d %f %f\n", roff, idx*sz_embed + roff % sz_embed, mpre[roff], mret[idx*sz_embed + roff % sz_embed], mpre[roff]);
		}
	} else {//msuf - input, mret - embeded, mpre - lookup table, roff는 mret 기준
		for(;roff < n; roff++) {
			int_val_type(idx, &msuf[roff / sz_embed], stp);
			mret[roff] = mpre[idx*sz_embed + roff % sz_embed];
		}
	}
	return n - idx_origin * idx_width;
}

class ApSoftmax : public Apply {
public:
	Flux *sfOut, *sfIn;
	Matrixr *sfmSum, *sfmMax;
	OprVar1 sfOprv;
	Flux *smcross, *smcross2, *a_dx, *a_dx2;
	Apply *cross_ap, *cross_ap2, *diag_ap, *diag_ap2;

	ApSoftmax(Trace *tcr, Flux *ip, Flux *op) : Apply(tcr)
	{
		apCode = APC_SOFTMAX;
		sfIn = ip;
		sfOut = op;
		nfanIn = 1;
		nfanOut = 1;
		sfmSum = sfmMax = nullx;

		registPhoto(ip, nullx);
		listingApin(ip);

		intt dim[MX_DIM], i = 0;
		for(;i < sfOut->fdim; i++) dim[i] = sfOut->fshape[i];
		dim[i] = dim[i - 1];
		tcr->directx(1);//그라프 연결하지 않게 한다.
		smcross = flux(tcr, sfOut->fdim + 1, dim, sfIn->qType, variable);
		smcross2 = *smcross * -1.0;//-aa

		a_dx = 1.0 - *sfOut;//(1-a)
		a_dx2 = *sfOut * *a_dx;//a(1-a)
		tcr->directx(0);//해제
	}
	void *forward(TContext *tcxt)
	{
		sytet iv = invariance2(sfIn);
		if(iv == 0) {
			sfOut->resizing2(sfIn);
			smcross->resizing2(sfIn);
		}
		if(iv <= 0) {
			sfmSum = apTcr->instMatrix(sfmSum, sfOut->qType, sfIn->fdim - 1, sfIn->fshape, false);
			sfmMax = apTcr->instMatrix(sfmMax, sfOut->qType, sfIn->fdim - 1, sfIn->fshape, false);
			sfOprv.noprDims1 = sfOut->fdim;
			memcpy(sfOprv.oprRank1, TENSOR(sfOut->quantum)->mxData->mxranksz, sizeof(intt) * sfOprv.noprDims1);
		}
		//printf("!!!!! SOFTMAX FORW \n");
		TENSOR(sfIn->quantum)->mxData->msoftmax(tcxt, TENSOR(sfOut->quantum)->mxData, sfmSum, sfmMax, &sfOprv);
		return sfOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		OneVar onevar;
		intt idf = sfOut->fdim - 1;
		//sfOut의 원래 shape은 [batch, seq, feat]이나 dimension 1은 어차피 크기가 없으므로  
		onevar.idxOne[0] = sfOut->fshape[idf];//M, //[batch, seq, feat, 1], mat mul하기위해 K값으로 1을 준다.
		onevar.idxOne[1] = 1;//K
		onevar.idxOne[2] = sfOut->fshape[idf];//N //[batch, seq, 1, feat]
		onevar.idxOne[3] = 0;
		TENSOR(sfOut->quantum)->mxData->mone(tcxt, TENSOR(sfOut->quantum)->mxData,
			TENSOR(smcross->quantum)->mxData, 2, &onevar, AOP_MATMUL, 0, PDG, 1);//[batch, seq, feat, feat]
		smcross2->exec(tcxt);//-aa [batch, seq, feat, feat]
		a_dx->exec(tcxt);//(1-a) [batch, seq, feat]
		a_dx2->exec(tcxt);//a(1-a) [batch, seq, feat]
		TENSOR(a_dx2->quantum)->mxData->mdiag_fill(nullx, TENSOR(smcross2->quantum)->mxData);//[[(1-a)a -aa]
																	//[batch, seq, feat, feat] [[-aa (1-a)a]]
		//delta-out([batch, seq, feat]) * delta-a([batch, seq, feat, feat])
		onevar.idxOne[0] = 1;//M, //[batch, seq, 1, feat], delta-out의 원래 shape은 [batch, seq, feat]이나 dimension 1은 어차피 크기가 없으므로 mat mul하기위해 M(로우)값으로 1을 준다.
		onevar.idxOne[1] = sfOut->fshape[idf];//K(feat)
		onevar.idxOne[2] = sfIn->fshape[idf];//N(feat) //[batch, seq, feat, feat]
		onevar.idxOne[3] = 0;//원래 출력 모양(delta-in)은 [batch, seq, 1, feat]가 되야하나 sfIn의 모양인 [batch, seq, feat]에 적재되도 된다. 1은 크기가 없으므로
		TENSOR(sfOut->quantum)->mxGrad->mone(tcxt, TENSOR(smcross2->quantum)->mxData,
			TENSOR(sfIn->quantum)->mxGrad, 2, &onevar, AOP_MATMUL, 0, PDG, 1);//delta-in([batch, seq, feat])
		return lapInput;
	}
};
class ApSoftmaxCrossE : public Apply {
public:
	Flux *sfcOut, *sfcIn, *sfcTar;
	Matrixr *sfmSum, *sfmMax, *sfomtx;
	ArithVar minusArv, devideArv;
	//unit withBatch;
	OneVar smcVar;

	ApSoftmaxCrossE(Trace *tcr, Flux *ip, Flux *tp, Flux *op) : Apply(tcr)
	{
		apCode = APC_SOFTCROSS;
		sfcIn = ip;
		sfcOut = op;
		sfcTar = tp;
		nfanIn = 2;
		nfanOut = 1;
		sfmSum = sfmMax = sfomtx = nullx;
		minusArv.paintVar = devideArv.paintVar = 0;

		registPhoto(ip, tp);
		listingApin(ip);
		listingApin(tp);//tp타겟은 역전파하지 않을 려면 현 라인과 asce), fsce)에서 fxt reference되지 않게 뺄것
	}
	void *forward(TContext *tcxt)
	{
		Flux *fxr;
		sytet iv = invariance(sfcIn, sfcTar, fxr);
		if(iv == 0) sfcOut->resizing2(fxr);
		if(iv <= 0) {
			sfmSum = apTcr->instMatrix(sfmSum, sfcOut->qType, sfcIn->fdim - 1, sfcIn->fshape, false);
			sfmMax = apTcr->instMatrix(sfmMax, sfcOut->qType, sfcIn->fdim - 1, sfcIn->fshape, false);
			sfomtx = apTcr->instMatrix(sfomtx, sfcOut->qType, sfcIn->fdim, sfcIn->fshape, false);
			minusArv.paintVar = devideArv.paintVar = 0;
			intt bn = (apTcr->ibatchSize ? apTcr->ibatchSize : sfcIn->fshape[0]);
			copy_cval_type(smcVar.idxOne, bn, sfomtx->mxType);//배치 사이즈
			//copy_cval_type(smcVar.idxOne, MTX_SIZE(sfmSum), sfomtx->mxType);//배치 사이즈
			//copy_cval_type(&withBatch, MTX_SIZE(sfmSum), sfomtx->mxType);//배치 사이즈
		}
		//printf("!!!!! SOFT_CROSS_E FORW \n");
		TENSOR(sfcIn->quantum)->mxData->msoftmax(tcxt, sfomtx, sfmSum, sfmMax, nullx);
		//sfomtx->printo();
		sfomtx->msoftx_cross_e(tcxt, TENSOR(sfcOut->quantum)->mxData, TENSOR(sfcTar->quantum)->mxData);
		//TENSOR(sfcOut->quantum)->mxData->printo();
		return sfcOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		//printf("!!!!! SOFT_CROSS_E BACK \n");
		bwLock(sfcIn, bw_mutmx);
		//sfomtx->marith(tcxt, TENSOR(sfcTar->quantum)->mxData, TENSOR(sfcIn->quantum)->mxGrad,
		//				&minusArv, 0, nullx, nullx, AOP_MINUS);
		//TENSOR(sfcIn->quantum)->mxGrad->marith(tcxt, nullx, TENSOR(sfcIn->quantum)->mxGrad,
		//	&devideArv, sfomtx->mxType, &withBatch, nullx, AOP_DIV);//배치 사이즈로 나눔.
		sfomtx->mone(tcxt, TENSOR(sfcTar->quantum)->mxData, TENSOR(sfcIn->quantum)->mxGrad,
			2, &smcVar, AOP_ACTF, DLOSS_FUNC, PDC, 1);
		bwUnlock(sfcIn);
		bwLock(sfcTar, bw_mutmx);
		//asce.이하. 타겟쪽 기울기 전파는 sgd는 정확하나 cross e는 오차가 발생, 뭔가 미분식이 틀린것 같음. 나중에 수정.
		//TENSOR(sfcTar->quantum)->mxData->marith(tcxt, sfomtx, TENSOR(sfcTar->quantum)->mxGrad,
		//	&minusArv, 0, nullx, nullx, AOP_MINUS);
		//TENSOR(sfcTar->quantum)->mxGrad->marith(tcxt, nullx, TENSOR(sfcTar->quantum)->mxGrad,
		//	&devideArv, sfomtx->mxType, &withBatch, nullx, AOP_DIV);//배치 사이즈로 나눔.
		TENSOR(sfcTar->quantum)->mxData->mone(tcxt, sfomtx, TENSOR(sfcTar->quantum)->mxGrad,
			2, &smcVar, AOP_ACTF, DLOSS_FUNC, PDC, 1);
		bwUnlock(sfcTar);
		return lapInput;
	}
};
void msoftmax(TContext *tcxt, Matrixr *rmtx, Matrixr *sum_mtx, Matrixr *max_mtx, OprVar1 *poprv)
{
	SignalR *sr = rsc::srutra->srGet();
	SoftmaxTrack<DT> *trc;
	OprVar1 *oprv;
	intt i, n, width;
	intt feat_sz = SZ_MTX_LOW_FIRST(rmtx);
	bool gpu = mxRutra->policyTrack(MTX_SIZE(rmtx), 1, feat_sz, PDG, width, n, mxTcr->cpmMode);

	if(poprv) {
		tcxt->cxbegin();
		oprv = (OprVar1 *)tcxt->cxalloc(sizeof(OprVar1), i);
		memcpy(oprv, poprv, sizeof(OprVar1));
		tcxt->syncCxt2Dev(gpu);
	}
	sum_mtx->resetMemory(gpu ? 1 : -1);
	max_mtx->resetMemory(gpu ? 1 : -1);
	SyncHstDev(gpu, 1, this, ((Matrix *)sum_mtx), ((Matrix *)max_mtx));//max_mtx가 리턴매트릭스가 아닌 읽기참조되는 것이므로 rcopy 1로 설정
	for(i = 0;i < n; i++) {
		trc = (SoftmaxTrack<DT> *)rsc::srutra->trkGet<SoftmaxTrack<DT>>(tcxt, i, width, gpu);
		trc->mSetSoftx(this, (Matrix<DT> *)rmtx, (Matrix<DT> *)sum_mtx, (Matrix<DT> *)max_mtx, feat_sz);
		trc->ontrack(sr, gpu);
	}
	sr->srWait();
	sr->srPut<SoftmaxTrack<DT>>();
}
template<typename DT> class SoftmaxTrack : public RootTrack {
public:
	static Tracker *trkPool;
	Matrix<DT> *mtxSrc, *mtxRet, *mtxSum, *mtxMax;
	intt sfxFeatsz;

	SoftmaxTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 5;
	}
	void mSetSoftx(Matrix<DT> *src, Matrix<DT> *ret, Matrix<DT> *sum, Matrix<DT> *max, intt fsz)
	{
		mtxSrc = src; mtxRet = ret; mtxSum = sum; mtxMax = max;
		sfxFeatsz = fsz;
	}
	void tracking(void)
	{//매트랙스들의 타입이 DT와 틀릴경우 DT와 틀린 매트릭스들을 나중에 캐스팅하여 처리한다.
		chron_begin(lap, mtxSrc);
		if(execGpu) {
			intt n = gsoftx_f(tcxtrTrk->mCxtDevice, mtxSrc->mxmDevice, mtxRet->mxmDevice, mtxSum->mxmDevice,
				mtxMax->mxmDevice, MTX_SIZE(mtxRet), sfxFeatsz,
				mtxRet->mxType < memput::mp::tlong ? 0 : 1, idxOrigin, widthPer);
			chron_end(lap, mtxSrc, mtxRet, "softmax", "gpu", n);
			CopyDeviceToHost2(mtxRet, idxOrigin * widthPer, n, 1);
			rutra->puthGpu(execGpu, n);
		} else {
			intt n = csoftx_t<DT>(tcxtrTrk->mCxtHost, mtxSrc->mxmHost, mtxRet->mxmHost, mtxSum->mxmHost,
				mtxMax->mxmHost, MTX_SIZE(mtxRet), sfxFeatsz, idxOrigin, widthPer);
			chron_end(lap, mtxSrc, mtxRet, "softmax", "cpu", n);
			CopyHostToDevice2(mtxRet, idxOrigin * widthPer, n, 1);
			rutra->puthGpu(nullx, n);
		}
		srGate->srReturn();
	}
};
template<typename DT> Tracker *SoftmaxTrack<DT>::trkPool = nullx;

template<typename T>
__device__ void datomic_max_f(T *pmax, const T value)
{
	intt * const imax = (int *)pmax;
	intt old = *imax, vcmp;

	while(value > __int_as_float(old)) {
		//printf("%f %f %d\n", value, __int_as_float(old), old);
		vcmp = old;
		old = atomicCAS(imax, vcmp, __float_as_int(value));
	}
}
template<typename T>
__global__ void ksoftx_prob_f(void *pcxt, T *m_rsfx, T *m_sum, intt f_size, intt idx_origin, intt idx_width, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	*(m_rsfx + roff) = *(m_rsfx + roff) / (*(m_sum + roff / f_size) + 1e-8);
}
template<typename T>
__global__ void ksoftx_sum_f(void *pcxt, T *m_ssfx, T *m_rsfx, T *m_sum, T *m_max, intt f_size, sytet db, intt idx_origin, intt idx_width, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;
	T v;
	if(roff >= n) return;

	if(db) v = std::exp((doublet)(*(m_ssfx + roff) - *(m_max + roff / f_size)));
	else v = std::exp((floatt)(*(m_ssfx + roff) - *(m_max + roff / f_size)));
	atomicAdd(m_sum + roff / f_size, v);
	*(m_rsfx + roff) = v;

}
template<typename T>
__global__ void ksoftx_max_f(void *pcxt, T *m_ssfx, T *m_max, intt f_size, intt idx_origin, intt idx_width, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	datomic_max_f(m_max + roff / f_size, *(m_ssfx + roff));
}
template<typename T>
intt gsoftx_t(void *pcxt, T *m_ssfx, T *m_rsfx, T *m_sum, T *m_max, intt r_size, intt f_size, sytet db, intt idx_origin, intt idx_width)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	ksoftx_max_f<T> << <grid, block >> > (pcxt, m_ssfx, m_max, f_size, idx_origin, idx_width, n);
	ksoftx_sum_f<T> << <grid, block >> > (pcxt, m_ssfx, m_rsfx, m_sum, m_max, f_size, db, idx_origin, idx_width, n);
	ksoftx_prob_f<T> << <grid, block >> > (pcxt, m_rsfx, m_sum, f_size, idx_origin, idx_width, n);
	cudaThreadSynchronize();
	cuda_error_check(-24);
	return n - idx_origin * idx_width;
}
intt gsoftx_f(void *pcxt, floatt *m_ssfx, floatt *m_rsfx, floatt *m_sum, floatt *m_max, intt r_size, intt f_size, sytet db, intt idx_origin, intt idx_width)
{
	return gsoftx_t<floatt>(pcxt, m_ssfx, m_rsfx, m_sum, m_max, r_size, f_size, db, idx_origin, idx_width);
}
intt gsoftx_f(void *pcxt, intt *m_ssfx, intt *m_rsfx, intt *m_sum, intt *m_max, intt r_size, intt f_size, sytet db, intt idx_origin, intt idx_width)
{
	return gsoftx_t<intt>(pcxt, m_ssfx, m_rsfx, m_sum, m_max, r_size, f_size, db, idx_origin, idx_width);
}