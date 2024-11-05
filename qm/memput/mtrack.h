#pragma once

#include "mcpu.h"
#include "mgpu.h"
#include "mcpu2.h"
#include "trace.h"

//#define BLOCK_SIZE 1024
//#define SMALL_BLOCK 512

#define MTX_SIZE_RK(mtx, irk) MRANK_SIZE(mtx->mxranksz, irk)
#define MTX_SIZE_RK2(mtx, irk) MRANK_SIZE2(mtx->mxranksz, irk)
#define MTX_SIZE(mtx) MTX_SIZE_RK(mtx, 0)
#define SZ_MTX_LOW_FIRST(mtx) MTX_SIZE_RK(mtx, mtx->mxndim - 1)
#define SZ_MTX_LOW_SECOND(mtx) MTX_SIZE_RK(mtx, mtx->mxndim - 2)
#define DIM_MTX_LOW_FIRST(mtx) mtx->mxshape[mtx->mxndim - 1]
#define DIM_MTX_LOW_SECOND(mtx) mtx->mxshape[mtx->mxndim - 2]

#define CopyHostToDevice(matx, host_exec) {\
	/*LOCK_MUT_(rsc::mutgpu);*/\
	if(matx->realtied == 0) {\
		if(host_exec) {\
			matx->cpmhot = 1;/*호스트 메모리가 변경된후 호출*/\
		} else if(matx->cpmhot > 0) {\
			/*matx->debugStamp("aa");*/\
			matx->cpmhot = 0;/*디바이스로 메모리 복사 위해 호출*/\
			cudaError_t error = cudaMemcpy(matx->mxmDevice, matx->mxmHost, MTX_SIZE(matx) * sizeof(DT), cudaMemcpyHostToDevice);\
			/*UNLOCK_MUT_(rsc::mutgpu);*/\
			if(error != cudaSuccess) throwFault(-1, "host to device copy error: %s\n", cudaGetErrorString(error));\
		}\
	} else if(host_exec) {\
		cudaError_t error = cudaMemcpy(matx->mxmDevice, matx->mxmHost, MTX_SIZE(matx) * sizeof(DT), cudaMemcpyHostToDevice);\
		/*UNLOCK_MUT_(rsc::mutgpu);*/\
		if(error != cudaSuccess) throwFault(-1, "host to device copy error: %s\n", cudaGetErrorString(error));\
	}\
}
#define CopyHostToDevice2(matx, base, size, host_exec) {\
	/*LOCK_MUT_(rsc::mutgpu);*/\
	if(matx->realtied == 0) {\
		if(host_exec) {\
			matx->cpmhot = 1;/*호스트 메모리가 변경된후 호출*/\
		} else if(matx->cpmhot > 0) {\
			/*matx->debugStamp("bb");*/\
			matx->cpmhot = 0;/*디바이스로 메모리 복사 위해 호출*/\
			cudaError_t error = cudaMemcpy(matx->mxmDevice + base, matx->mxmHost + base, size * sizeof(DT), cudaMemcpyHostToDevice);\
			/*UNLOCK_MUT_(rsc::mutgpu);*/\
			if(error != cudaSuccess) throwFault(-1, "host to device2 copy error: %s\n", cudaGetErrorString(error));\
		}\
	} else if(host_exec) {\
		cudaError_t error = cudaMemcpy(matx->mxmDevice + base, matx->mxmHost + base, size * sizeof(DT), cudaMemcpyHostToDevice);\
		/*UNLOCK_MUT_(rsc::mutgpu);*/\
		if(error != cudaSuccess) throwFault(-1, "host to device2 copy error: %s\n", cudaGetErrorString(error));\
	}\
}
#define CopyDeviceToHost(matx, dev_exec) {\
	/*LOCK_MUT_(rsc::mutgpu);*/\
	if(matx->realtied == 0) {\
		if(dev_exec) matx->cpmhot = -1;\
		else if(matx->cpmhot < 0) {\
			/*matx->debugStamp("cc");*/\
			matx->cpmhot = 0;\
			cudaError_t error = cudaMemcpy(matx->mxmHost, matx->mxmDevice, MTX_SIZE(matx) * sizeof(DT), cudaMemcpyDeviceToHost);\
			/*UNLOCK_MUT_(rsc::mutgpu);*/\
			if(error != cudaSuccess) throwFault(-1, "device to host copy error: %s\n", cudaGetErrorString(error));\
		}\
	} else if(dev_exec) {\
		cudaError_t error = cudaMemcpy(matx->mxmHost, matx->mxmDevice, MTX_SIZE(matx) * sizeof(DT), cudaMemcpyDeviceToHost);\
		/*UNLOCK_MUT_(rsc::mutgpu);*/\
		if(error != cudaSuccess) throwFault(-1, "device to host copy error: %s\n", cudaGetErrorString(error));\
	}\
}
#define CopyDeviceToHost2(matx, base, size, dev_exec) {\
	/*LOCK_MUT_(rsc::mutgpu);*/\
	if(matx->realtied == 0) {\
		if(dev_exec) matx->cpmhot = -1;\
		else if(matx->cpmhot < 0) {\
			/*matx->debugStamp("dd");*/\
			matx->cpmhot = 0;\
			cudaError_t error = cudaMemcpy(matx->mxmHost + base, matx->mxmDevice + base, size * sizeof(DT), cudaMemcpyDeviceToHost);\
			/*UNLOCK_MUT_(rsc::mutgpu);*/\
			if(error != cudaSuccess) throwFault(-1, "device to host2 copy error: %s\n", cudaGetErrorString(error));\
		}\
	} else if(dev_exec) {\
		cudaError_t error = cudaMemcpy(matx->mxmHost + base, matx->mxmDevice + base, size * sizeof(DT), cudaMemcpyDeviceToHost);\
		/*UNLOCK_MUT_(rsc::mutgpu);*/\
		if(error != cudaSuccess) throwFault(-1, "device to host2 copy error: %s\n", cudaGetErrorString(error));\
	}\
}
#define SyncHstDev(gpu, rcopy, pre, suf, ret) {\
	if(gpu) {\
		CopyHostToDevice(pre, 0);\
		if(suf) CopyHostToDevice(suf, 0);\
		if(rcopy) CopyHostToDevice(ret, 0);\
	} else {\
		CopyDeviceToHost(pre, 0);\
		if(suf) CopyDeviceToHost(suf, 0);\
		if(rcopy) CopyDeviceToHost(ret, 0);\
	}\
}

class Matrixr;
template<typename DT> class Matrix;

class RootTrack : public Tracker {
public:
	intt idxOrigin, widthPer;
	TContext *tcxtrTrk;

	void mSetRtra(void *tcxt, intt iori, intt width, CudaHandle *hgpu)
	{
		tcxtrTrk = (TContext *)tcxt;
		idxOrigin = iori;
		widthPer = width;
		gpuSetting(tcxtrTrk->ground_did, hgpu);
	}
	RootTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : Tracker(rt)
	{
		mSetRtra(tcxt, iori, width, hgpu);
	}
};
template<typename DT> class ConcatTrack : public RootTrack {
public:
	static Tracker *trkPool;
	Matrix<DT> *mtxPrimary, **mtxSecondary;
	intt nPartCT, nStepCT, axisCT, sdimCT;
	bool ctConcat, bwCatra;

	ConcatTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 1;
	}
	void mSetCatra(Matrix<DT> *primary, intt nsplit, intt nstep, intt axis, intt sdim, bool concat, void *psplit_mtx, bool bw)
	{
		mtxPrimary = primary;
		nPartCT = nsplit;
		nStepCT = nstep;
		axisCT = axis;
		sdimCT = sdim;
		ctConcat = concat;
		mtxSecondary = (Matrix<DT> **)psplit_mtx;
		bwCatra = bw;
	}
	void secondaryCopy(intt n, bool h2d, void *pcxt, intt pdim, intt nsplit, intt nstep, intt axis)
	{//비 균등 분할 추가로 복잡성이 커 사요 안함.
		ConcatVar *ccv = (ConcatVar *)pcxt;

		intt *prank = P_LINK_VAR2(intt, pcxt, ccv->szRankPrimary);
		intt outer_sz = MRANK_SIZE(prank, axis), inner_sz = (axis == pdim - 1 ? 1 : MRANK_SIZE(prank, axis + 1)) * nstep;
		intt roff, soff, oi, si, base, wsz, rest;

		roff = idxOrigin * widthPer;
		oi = roff / outer_sz;
		soff = roff % outer_sz;//split축 랭크내의 분할 옵셋
		si = soff / inner_sz;//split축 분할 단위 이번 분할번째
		base = oi * inner_sz + soff % inner_sz;//분할번째 시작 옵셋
		wsz = inner_sz - soff % inner_sz;
		if(wsz > n) wsz = n;
		for(rest = n;wsz;) {
			if(h2d) {
				CopyHostToDevice2((*(mtxSecondary + si)), base, wsz, 1);
			} else CopyDeviceToHost2((*(mtxSecondary + si)), base, wsz, 1);
			if(++si == nsplit) {
				si = 0;
				oi++;
			}
			base = oi * inner_sz;
			rest -= wsz;
			if(rest < inner_sz) wsz = rest;
			else wsz = inner_sz;
		}
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		intt n;
		chron_begin(lap, mtxPrimary);
		if(execGpu) {//gpu로 실행할지를 결정하는 용도로만 사용.
			if(ctConcat) {
				n = gconcat_f(tcxtrTrk->mCxtDevice, mtxPrimary->mxmDevice, mtxPrimary->mxndim, sdimCT, MTX_SIZE(mtxPrimary),
					idxOrigin, widthPer, nPartCT, nStepCT, axisCT, bwCatra);
				chron_end(lap, mtxPrimary, mtxPrimary, "concat", "cpu", n);
				CopyDeviceToHost2(mtxPrimary, idxOrigin * widthPer, n, 1);
				rutra->puthGpu(execGpu, n);//kkk
			} else {
				n = gsplit_f(tcxtrTrk->mCxtDevice, mtxPrimary->mxmDevice, mtxPrimary->mxndim, sdimCT, MTX_SIZE(mtxPrimary),
					idxOrigin, widthPer, nPartCT, nStepCT, axisCT, bwCatra);
				chron_end(lap, mtxPrimary, mtxPrimary, "split", "gpu", n);
				srGate->checkSide(true);
				//secondaryCopy(n, false, tcxtrTrk->mCxtHost, mtxPrimary->mxndim, nPartCT, nStepCT, axisCT);
				rutra->puthGpu(execGpu, n);
			}
		} else {
			if(ctConcat) {
				n = cconcat_t<DT>(tcxtrTrk->mCxtHost, mtxPrimary->mxmHost, mtxPrimary->mxndim, sdimCT, MTX_SIZE(mtxPrimary),
					idxOrigin, widthPer, nPartCT, nStepCT, axisCT, bwCatra);
				chron_end(lap, mtxPrimary, mtxPrimary, "concat", "cpu", n);
				CopyHostToDevice2(mtxPrimary, idxOrigin * widthPer, n, 1);
				rutra->puthGpu(nullx, n);
			} else {
				n = csplit_t<DT>(tcxtrTrk->mCxtHost, mtxPrimary->mxmHost, mtxPrimary->mxndim, sdimCT, MTX_SIZE(mtxPrimary),
					idxOrigin, widthPer, nPartCT, nStepCT, axisCT, bwCatra);
				chron_end(lap, mtxPrimary, mtxPrimary, "split", "cpu", n);
				srGate->checkSide(false);
				//secondaryCopy(n, true, tcxtrTrk->mCxtHost, mtxPrimary->mxndim, nPartCT, nStepCT, axisCT);
				rutra->puthGpu(nullx, n);
			}
		}
		srGate->srReturn();
	}
};
template<typename DT> Tracker *ConcatTrack<DT>::trkPool = nullx;

template<typename DT> class ArithTrack : public RootTrack {
public:
	static Tracker *trkPool;
	Matrix<DT> *mtxPref, *mtxSuf, *mtxRet;
	DT scalarV, rPlusV;
	ubytet tscalV;
	sytet opArith;
	bool onlyCpu;

	ArithTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 2;
	}
	void mSetAtra(Matrix<DT> *pref, Matrix<DT> *suf, Matrix<DT> *ret, ubytet tval, DT sval, sytet aop, DT rplus)
	{
		mtxPref = pref; mtxSuf = suf; mtxRet = ret;
		tscalV = tval;
		scalarV = sval;
		opArith = aop;
		rPlusV = rplus;
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		if(tcxtrTrk) {
			ArithVar *arv = (ArithVar *)tcxtrTrk->mCxtHost;
			chron_begin2(lap, t, mtxPref);
			//chrono::system_clock::time_point lap = chrono::system_clock::now();//(mtxPref->lapType == 1 ? chrono::system_clock::now() : 0);
			if(execGpu) {//gpu로 실행할지를 결정하는 용도로만 사용.
				intt n = garith_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, (mtxSuf ? mtxSuf->mxmDevice : nullx),
					mtxRet->mxmDevice, MTX_SIZE(mtxPref), MTX_SIZE(mtxRet), idxOrigin,
					widthPer, scalarV, opArith, rPlusV, arv->tpArith, arv->bwGetOri);
				chron_end2(lap, mtxPref, mtxSuf, mtxRet, "arith", "gpu", n, t, arv->tpArith, arv->bwGetOri, opArith);
				CopyDeviceToHost2(mtxRet, idxOrigin * widthPer, n, 1);
				rutra->puthGpu(execGpu, n);
			} else {
				intt n;// = carith_t<DT>(tcxtrTrk->mCxtHost, (mtxPref == mtxSuf && tscalV ? nullx : mtxPref->mxmHost),
					//(mtxSuf ? mtxSuf->mxmHost : nullx), mtxRet->mxmHost, MTX_SIZE(mtxRet), idxOrigin,
					//widthPer, scalarV, opArith, rPlusV);
				switch(arv->tpArith) {
				case AR_T_O2O:
					n = carith_t1<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf->mxmHost, mtxRet->mxmHost,
						MTX_SIZE(mtxRet), idxOrigin, widthPer, opArith, rPlusV);
					break;
				case AR_T_BROLC:
					n = carith_t2_lc<DT>(tcxtrTrk->mCxtHost, arv->bwGetOri ? mtxPref->mxmHost : mtxSuf->mxmHost,
						mtxRet->mxmHost, MTX_SIZE(mtxRet), idxOrigin, widthPer, scalarV, opArith, rPlusV);
					break;
				case AR_T_BRORC:
					n = carith_t2_rc<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxRet->mxmHost, MTX_SIZE(mtxRet),
						idxOrigin, widthPer, scalarV, opArith, rPlusV);
					break;
				case AR_T_BRO:
					n = carith_t2<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf->mxmHost, mtxRet->mxmHost,
						MTX_SIZE(mtxRet), idxOrigin, widthPer, opArith, rPlusV);
					break;
				case AR_T_ONEBRO:
					n = carith_t3<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf->mxmHost, mtxRet->mxmHost,
						MTX_SIZE(mtxRet), idxOrigin, widthPer, opArith, rPlusV);
					break;
				}
				chron_end(lap, mtxPref, mtxRet, "arith", "cpu", n);
				CopyHostToDevice2(mtxRet, idxOrigin * widthPer, n, 1);//pref와 suff가 같으면 pref가 scalarV이다.
				rutra->puthGpu(nullx, n);
			}
		}
		srGate->srReturn();
	}
};
template<typename DT> Tracker *ArithTrack<DT>::trkPool = nullx;

template<typename DT> class DotTrack : public RootTrack {
public:
	static Tracker *trkPool;
	Matrix<DT> *mtxPref, *mtxSuf, *mtxRet;
	DotVar *dtGstr;
	DT rPlus;
	sytet ordTrans;
	bool dot_t_1;

	DotTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 3;
	}
	void mSetDtra(Matrix<DT> *pref, Matrix<DT> *suf, Matrix<DT> *ret, bool dot_t, DT rplus)
	{
		mtxPref = pref; mtxSuf = suf; mtxRet = ret;
		rPlus = rplus;
		dot_t_1 = dot_t;
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		chron_begin(lap, mtxPref);
		if(execGpu <= (CudaHandle *)1) {
			DotVar *dotv = (DotVar *)tcxtrTrk->mCxtHost;
			intt n;
			if(execGpu) {//gpu로 실행할지를 결정하는 용도로만 사용.
				if(dot_t_1) n = gdot_f(tcxtrTrk->mCxtDevice, dotv->szOutKernel, dotv->shareUnit, mtxPref->mxmDevice, mtxSuf->mxmDevice, mtxRet->mxmDevice, MTX_SIZE(mtxRet),
					idxOrigin, widthPer, rPlus);
				else n = gdot_f2(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxSuf->mxmDevice, mtxRet->mxmDevice, MTX_SIZE(mtxRet),
					idxOrigin, widthPer, rPlus);
				chron_end(lap, mtxPref, mtxRet, "dot", "gpu", n);
				if(dotv->intervOut) srGate->checkSide(true);//위 함수에서 좌표 변환되어 출력되므로 리턴해서 전체 범위를 한꺼번에 오버라이트한다.
															//dot version 1도 백워드 실행일 경우 좌표 변환되므로 나주에 사용할 경우 검토한다.
				else CopyDeviceToHost2(mtxRet, idxOrigin * widthPer, n, 1);
				rutra->puthGpu(execGpu, n);
			} else {
				//unit lap2 = xucurrenttime();
				if(dot_t_1) n = cdot_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf->mxmHost, mtxRet->mxmHost, MTX_SIZE(mtxRet), idxOrigin, widthPer, rPlus);
				else n = cdot_t2<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf->mxmHost, mtxRet->mxmHost, MTX_SIZE(mtxRet), idxOrigin, widthPer, rPlus);
				//printf("xx %d lap: %f\n", n, (xucurrenttime() - lap2) / 1000000.0);
				chron_end(lap, mtxPref, mtxRet, "dot", "cpu", n);
				if(dotv->intervOut) srGate->checkSide(false);//위 설명과 동일
				else CopyHostToDevice2(mtxRet, idxOrigin * widthPer, n, 1);
				rutra->puthGpu(nullx, n);
			}
		} else {//cublas는 분할실행할수없다. suf N사이즈 단위로해서 pre M갯수로 분할해야 하는데 전치될때 ret매트릭스 기준으로
			//분할하면 suf N사이즈 단위로 끊어지지 않을수있고 분할할때 pre매트릭스 기준으로 분할한다면 roff가 분할간에 겹칠수있다.
			intt n = MTX_SIZE(mtxRet);
			//if(execGpu) {
			/*printf("cublas gid %d %d %d %d trk_type %d\n", gidTrc, mtxPref->didFocus, mtxSuf->didFocus, mtxRet->didFocus, trkType);
			if(mtxPref->didFocus != gidTrc || mtxSuf->didFocus != gidTrc || mtxRet->didFocus != gidTrc) {
				printf("cublas consistance error\n", gidTrc);
				exit(1);
			}*/
			mdot2d<DT>(execGpu->hCuda, ordTrans & TOA, ordTrans & TOB, dtGstr->prem,
				dtGstr->sufn, dtGstr->joik, 1, mtxPref->mxmDevice, dtGstr->lda,
				mtxSuf->mxmDevice, dtGstr->ldb, rPlus, mtxRet->mxmDevice, dtGstr->ldc,
				mtxRet->mxType < memput::mp::tlong ? 0 : 1, false);
			chron_end(lap, mtxPref, mtxRet, "dot", "gpu", n);
			CopyDeviceToHost(mtxRet, 1);
			rutra->puthGpu(execGpu, n);
			//} else {//cublas실행은 무조건 gpu로만 하게했으므로 주석처리
			//	mdot2d<DT>(nullx, ordTrans & TOA, ordTrans & TOB, dtGstr->prem,
			//		dtGstr->sufn, dtGstr->joik, 1, mtxPref->mxmHost, dtGstr->lda,
			//		mtxSuf->mxmHost, dtGstr->ldb, rPlus, mtxRet->mxmHost, dtGstr->ldc, -1, true);
			//	CopyHostToDevice(mtxRet, 1);
			//	rutra->puthGpu(nullx, n);
			//}
		}
		srGate->srReturn();
	}

};
template<typename DT> Tracker *DotTrack<DT>::trkPool = nullx;

template<typename DT> class TransposeTrack : public RootTrack {
public:
	static Tracker *trkPool;
	bool bwTrans;
	Matrix<DT> *mtxSrc, *mtxRet;

	TransposeTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 4;
	}
	void mSetTrans(Matrix<DT> *src, Matrix<DT> *ret, bool bw)
	{
		mtxSrc = src; mtxRet = ret;
		bwTrans = bw;
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		chron_begin(lap, mtxSrc);
		if(execGpu) {//gpu로 실행할지를 결정하는 용도로만 사용.
			intt n = gtrans_f(tcxtrTrk->mCxtDevice, mtxSrc->mxmDevice, mtxRet->mxmDevice,
				MTX_SIZE(mtxRet), idxOrigin, widthPer, bwTrans);
			chron_end(lap, mtxSrc, mtxRet, "transpose", "gpu", n);
			CopyDeviceToHost2(mtxRet, idxOrigin * widthPer, n, 1);
			rutra->puthGpu(execGpu, n);
		} else {
			intt n = ctrs_t<DT>(tcxtrTrk->mCxtHost, mtxSrc->mxmHost, mtxRet->mxmHost, MTX_SIZE(mtxRet), idxOrigin, widthPer, bwTrans);
			chron_end(lap, mtxSrc, mtxRet, "transpose", "cpu", n);
			CopyHostToDevice2(mtxRet, idxOrigin * widthPer, n, 1);
			rutra->puthGpu(nullx, n);
		}
		srGate->srReturn();
	}
};
template<typename DT> Tracker *TransposeTrack<DT>::trkPool = nullx;

template<typename DT> class SoftmaxTrack : public RootTrack {
public:
	static Tracker *trkPool;
	Matrix<DT> *mtxSrc, *mtxRet, *mtxSum, *mtxMax, *mtxBuf;
	intt sfxFeatsz;

	SoftmaxTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 5;
	}
	void mSetSoftx(Matrix<DT> *src, Matrix<DT> *ret, Matrix<DT> *sum, Matrix<DT> *max, Matrix<DT> *mbuf, intt fsz)
	{
		mtxSrc = src; mtxRet = ret; mtxSum = sum; mtxMax = max, mtxBuf = mbuf;
		sfxFeatsz = fsz;
	}
	void tracking(TCxtArray *tcxt_arr)
	{//매트랙스들의 타입이 DT와 틀릴경우 DT와 틀린 매트릭스들을 나중에 캐스팅하여 처리한다.
		chron_begin(lap, mtxSrc);
		if(execGpu) {
			intt n = gsoftx_f(tcxtrTrk->mCxtDevice, mtxSrc->mxmDevice, mtxRet->mxmDevice, mtxSum->mxmDevice,
				mtxMax->mxmDevice, mtxBuf->mxmDevice, MTX_SIZE(mtxRet), sfxFeatsz,
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

template<typename DT> class SoftmaxCrossETrack : public RootTrack {
public:
	static Tracker *trkPool;
	Matrix<DT> *mtxSrc, *mtxRet, *mtxTar;
	intt sfxFeatsz;

	SoftmaxCrossETrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 6;
	}
	void mSetTrack(Matrix<DT> *src, Matrix<DT> *ret, void *tar, intt fsz)
	{
		mtxSrc = src; mtxRet = ret; mtxTar = (Matrix<DT> *)tar;
		sfxFeatsz = fsz;
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		chron_begin(lap, mtxSrc);
		if(execGpu) {
			intt n = gsoftx_cross_e_f(tcxtrTrk->mCxtDevice, mtxSrc->mxmDevice, mtxRet->mxmDevice, mtxTar->mxmDevice,
				MTX_SIZE(mtxSrc), sfxFeatsz, idxOrigin, widthPer);
			chron_end(lap, mtxSrc, mtxRet, "cross entrophy", "gpu", n);
			CopyDeviceToHost2(mtxRet, idxOrigin * (widthPer / sfxFeatsz), n / sfxFeatsz, 1);
			rutra->puthGpu(execGpu, n);
		} else {
			intt n = csoftx_cross_e_t(tcxtrTrk->mCxtHost, mtxSrc->mxmHost, mtxRet->mxmHost, mtxTar->mxmHost,
				MTX_SIZE(mtxSrc), sfxFeatsz, idxOrigin, widthPer);
			chron_end(lap, mtxSrc, mtxRet, "cross entrophy", "cpu", n);
			CopyHostToDevice2(mtxRet, idxOrigin * (widthPer / sfxFeatsz), n / sfxFeatsz, 1);
			rutra->puthGpu(nullx, n);
		}
		srGate->srReturn();
	}
};
template<typename DT> Tracker *SoftmaxCrossETrack<DT>::trkPool = nullx;

template<typename DT> class MeanSquareETrack : public RootTrack {
public:
	static Tracker *trkPool;
	Matrix<DT> *mtxSrc, *mtxTar, *mtxRet;
	bool withMean;
	MeanSquareETrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 7;
	}
	void mSetTrack(Matrix<DT> *src, Matrix<DT> *tar, Matrix<DT> *ret, bool mean)
	{
		mtxSrc = src; mtxTar = tar; mtxRet = ret;
		withMean = mean;
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		intt b, n;
		chron_begin(lap, mtxSrc);
		b = withMean ? (idxOrigin * widthPer) : 0;
		if(execGpu) {
			n = gmse_f(tcxtrTrk->mCxtDevice, mtxSrc->mxmDevice, mtxTar->mxmDevice, mtxRet->mxmDevice, MTX_SIZE(mtxSrc), idxOrigin, widthPer, withMean);
			chron_end(lap, mtxSrc, mtxRet, "mse", "gpu", n);//gpu 실행은 사이즈가 gpu max를 초과할 경우 분할 실행됨.
			CopyDeviceToHost2(mtxRet, b, n, 1);//src를 기준으로 하여 나중에 한꺼번에 해야하나 타겟 1개에 집중되므로 여기서 여러번 복사해도 상관없다.
		} else {
			n = cmse_f(tcxtrTrk->mCxtHost, mtxSrc->mxmHost, mtxTar->mxmHost, mtxRet->mxmHost, MTX_SIZE(mtxSrc), idxOrigin, widthPer, withMean);
			chron_end(lap, mtxSrc, mtxRet, "mse", "cpu", n);//cpu실행은 분할 실행 없으므로 전채 사이즈 반납
			CopyHostToDevice2(mtxRet, b, n, 1);
		}
		rutra->puthGpu(nullx, n);
		srGate->srReturn();
	}
};
template<typename DT> Tracker *MeanSquareETrack<DT>::trkPool = nullx;

template<typename DT> class SumTrack : public RootTrack {
public:
	static Tracker *trkPool;
	Matrix<DT> *mtxSrc, *mtxRet;
	DT *cmulST;
	bool meanST;
	SumTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 8;
	}
	void mSetTrack(Matrix<DT> *src, void *ret, void *cmul, bool mean)
	{
		mtxSrc = src; mtxRet = (Matrix<DT> *)ret;
		cmulST = (DT *)cmul;
		meanST = mean;
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		intt n;
		chron_begin(lap, mtxSrc);
		if(execGpu) {
			n = gsum_f(tcxtrTrk->mCxtDevice, mtxSrc->mxmDevice, mtxRet->mxmDevice, MTX_SIZE(mtxSrc), idxOrigin, widthPer, cmulST, meanST);
			chron_end(lap, mtxSrc, mtxRet, "sum", "gpu", n);//gpu 실행은 사이즈가 gpu max를 초과할 경우 분할 실행됨.
			CopyDeviceToHost2(mtxRet, 0, 1, 1);//src를 기준으로 하여 나중에 한꺼번에 해야하나 타겟 1개에 집중되므로 여기서 여러번 복사해도 상관없다.
		} else {
			n = csum_f(tcxtrTrk->mCxtHost, mtxSrc->mxmHost, mtxRet->mxmHost, MTX_SIZE(mtxSrc), cmulST, meanST);
			chron_end(lap, mtxSrc, mtxRet, "sum", "cpu", n);//cpu실행은 분할 실행 없으므로 전채 사이즈 반납
			CopyHostToDevice2(mtxRet, 0, 1, 1);

		}
		rutra->puthGpu(nullx, n);
		srGate->srReturn();
	}
};
template<typename DT> Tracker *SumTrack<DT>::trkPool = nullx;

template<typename DT> class OptAdmTrack : public RootTrack {
public:
	static Tracker *trkPool;
	floatt beta1, beta2, lr, ep;
	intt dec;
	Matrix<DT> *mmtx, *vmtx, *gmtx, *rmtx;

	OptAdmTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 9;
	}
	void mSetTrack(void *m_mtx, void *v_mtx, void *g_mtx, void *r_mtx, floatt b1, floatt b2, floatt r, floatt e, intt d)
	{
		mmtx = (Matrix<DT> *)m_mtx; vmtx = (Matrix<DT> *)v_mtx; gmtx = (Matrix<DT> *)g_mtx;
		rmtx = (Matrix<DT> *)r_mtx; beta1 = b1; beta2 = b2; lr = r; ep = e; dec = d;
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		chron_begin(lap, mmtx);
		if(execGpu) {//gpu로 실행할지를 결정하는 용도로만 사용.
			intt n = goptadm_f(tcxtrTrk->mCxtDevice, mmtx->mxmDevice, vmtx->mxmDevice, gmtx->mxmDevice,
				rmtx->mxmDevice, MTX_SIZE(rmtx), idxOrigin, widthPer, beta1, beta2, lr, ep, dec,
				rmtx->mxType < memput::mp::tlong ? 0 : 1);
			chron_end(lap, mmtx, rmtx, "adam", "gpu", n);
			CopyDeviceToHost2(mmtx, idxOrigin * widthPer, n, 1);
			CopyDeviceToHost2(vmtx, idxOrigin * widthPer, n, 1);
			CopyDeviceToHost2(rmtx, idxOrigin * widthPer, n, 1);
			rutra->puthGpu(execGpu, n);
		} else {
			intt n = coptadm_t<DT>(tcxtrTrk->mCxtHost, mmtx->mxmHost, vmtx->mxmHost, gmtx->mxmHost,
				rmtx->mxmHost, MTX_SIZE(rmtx), idxOrigin, widthPer, beta1, beta2, lr, ep, dec);
			chron_end(lap, mmtx, rmtx, "adam", "cpu", n);
			CopyHostToDevice2(mmtx, idxOrigin * widthPer, n, 1);
			CopyHostToDevice2(vmtx, idxOrigin * widthPer, n, 1);
			CopyHostToDevice2(rmtx, idxOrigin * widthPer, n, 1);
			rutra->puthGpu(nullx, n);
		}
		srGate->srReturn();
	}
};
template<typename DT> Tracker *OptAdmTrack<DT>::trkPool = nullx;

template<typename DT> class OptSgdTrack : public RootTrack {
public:
	static Tracker *trkPool;
	floatt beta1, beta2, lr, ep;
	intt dec;
	Matrix<DT> *gmtx, *rmtx;

	OptSgdTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 10;
	}
	void mSetTrack(Matrix<DT> *g_mtx, void *r_mtx, floatt r, intt d)
	{
		gmtx = g_mtx; rmtx = (Matrix<DT> *)r_mtx; lr = r; dec = d;
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		chron_begin(lap, gmtx);
		if(execGpu) {//gpu로 실행할지를 결정하는 용도로만 사용.
			intt n = goptsgd_f(tcxtrTrk->mCxtDevice, gmtx->mxmDevice, rmtx->mxmDevice, MTX_SIZE(rmtx),
				idxOrigin, widthPer, lr, dec);
			chron_end(lap, gmtx, rmtx, "sgd", "gpu", n);
			CopyDeviceToHost2(rmtx, idxOrigin * widthPer, n, 1);
			rutra->puthGpu(execGpu, n);
		} else {
			intt n = coptsgd_t<DT>(tcxtrTrk->mCxtHost, gmtx->mxmHost, rmtx->mxmHost, MTX_SIZE(rmtx),
				idxOrigin, widthPer, lr, dec);
			chron_end(lap, gmtx, rmtx, "sgd", "cpu", n);
			CopyHostToDevice2(rmtx, idxOrigin * widthPer, n, 1);
			rutra->puthGpu(nullx, n);
		}
		srGate->srReturn();
	}
};
template<typename DT> Tracker *OptSgdTrack<DT>::trkPool = nullx;

#define AOP_ACTF	0
#define AOP_ACTF2	1
#define AOP_EMBEDDING	2
#define AOP_SLICE	3
#define AOP_ONEHOT	4
#define AOP_ARGMAX	5
#define AOP_EQUAL	6
#define AOP_TYPE1	7
#define AOP_RANDOM	8
#define AOP_LAYNOR	9
#define AOP_MATMUL	10
#define AOP_BSUM	11
#define AOP_VMAX	12

template<typename DT> class OneTrack : public RootTrack {
public:
	static Tracker *trkPool;
	intt aopTrk, aopTrk2, primesz;
	sytet rplusOk;
	Matrix<DT> *mtxPref, *mtxSuf, *mtxRet, *mxrsuf;
	OneVar *ovar;
	DT **p;

	OneTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 11;
	}
	void mSetTrack(Matrix<DT> *p_mtx, Matrixr *s_mtx, Matrix<DT> *r_mtx, intt aop, intt aop2, intt sz, sytet rplus)
	{
		mtxPref = p_mtx; mtxSuf = (Matrix<DT> *)s_mtx; mtxRet = r_mtx; aopTrk = aop; aopTrk2 = aop2;
		primesz = sz;
		rplusOk = rplus;
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		intt n;
		chron_begin(lap, mtxPref);
		if(execGpu) {//gpu로 실행할지를 결정하는 용도로만 사용.
			switch(aopTrk) {
			case AOP_ACTF:
				n = gactf_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxSuf ? mtxSuf->mxmDevice : nullx, mtxRet->mxmDevice, primesz,
					idxOrigin, widthPer, aopTrk2, rplusOk, mtxRet->mxType < memput::mp::tlong ? 0 : 1);
				break;
			case AOP_ACTF2:
			{
				ovar = (OneVar *)tcxtrTrk->mCxtHost;
				Matrix<DT> *m1 = *(Matrix<DT> **)ovar->idxOne;
				Matrix<DT> *m2 = *(Matrix<DT> **)&ovar->idxOne[2];
				n = gactf2_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxSuf->mxmDevice, mtxRet->mxmDevice,
					m1->mxmDevice, m2->mxmDevice, primesz, idxOrigin, widthPer, aopTrk2, rplusOk, mtxRet->mxType < memput::mp::tlong ? 0 : 1);
				//m2도 갱신되므로 copy host memory해야하나 m2는 이 함수에서만 사용되므로 모드간 
				//전환되어 엑세스될 일 없으므로 하지 않는다.
			}
			break;
			case AOP_EMBEDDING:
				n = gembedding_f(mtxPref->mxmDevice, mtxSuf->mxmDevice, mtxRet->mxmDevice, primesz,
					idxOrigin, widthPer, SZ_MTX_LOW_FIRST(mtxPref), mtxSuf->mxType,
					aopTrk2 ? MTX_SIZE(mtxRet) : MTX_SIZE(mtxPref), aopTrk2);
				n *= -1;
				//srGate->checkSide(true);//역전파일만 pref를 기준으로 하여 순전파일때는 여기서 복사해도 되나 일관성을 위해 순전파도 후 복사한다.
				break;
			case AOP_SLICE:
				n = gslice_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxRet->mxmDevice, primesz,
					idxOrigin, widthPer, aopTrk2, rplusOk);
				n *= -1;
				//srGate->checkSide(true);//pref를 기준으로하므로 후 복사한다.
				break;
			case AOP_ONEHOT:
				n = gonehot_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxRet->mxmDevice, primesz,
					idxOrigin, widthPer);
				n *= -1;
				//srGate->checkSide(true);//pref를 기준으로하므로 후 복사한다.
				break;
			case AOP_ARGMAX:
				n = gargmax_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxRet->mxmDevice, primesz,
					idxOrigin, widthPer);
				break;
			case AOP_EQUAL:
				n = gequal_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxSuf ? mtxSuf->mxmDevice : nullx, mtxRet->mxmDevice, primesz,
					idxOrigin, widthPer);
				break;
			case AOP_TYPE1:
				n = gtype1_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxSuf ? mtxSuf->mxmDevice : nullx,
					mtxRet ? mtxRet->mxmDevice : nullx, primesz, idxOrigin, widthPer, aopTrk2);
				break;
			case AOP_RANDOM:
				n = grandom_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, primesz, idxOrigin, widthPer, aopTrk2, tcxtrTrk->tcxttrc->rseed);
				break;
			case AOP_LAYNOR:
				p = (DT **)tcxtrTrk->mCxtHost;
				n = glayer_norm_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxRet->mxmDevice, *p,
					*(p + 1), *(p + 2), *(p + 3), *(p + 4), *(p + 5), *(p + 6), *(p + 7),
					*(p + 8), *(p + 9), *(p + 10), primesz, idxOrigin, widthPer, aopTrk2, rplusOk,
					mtxRet->mxType < memput::mp::tlong ? 0 : 1);
				n *= -1;
				//srGate->checkSide(true);
				break;
			case AOP_MATMUL:
				ovar = (OneVar *)tcxtrTrk->mCxtHost;
				n = gmatmul_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxSuf->mxmDevice, mtxRet->mxmDevice,
					primesz, ovar->idxOne[0], ovar->idxOne[1], ovar->idxOne[2], ovar->idxOne[3], rplusOk,
					idxOrigin, widthPer);
				break;
			case AOP_BSUM:
				ovar = (OneVar *)tcxtrTrk->mCxtHost;
				n = gbsum_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxRet->mxmDevice, primesz,
					idxOrigin, widthPer, ovar->idxOne[1], ovar->idxOne[0], aopTrk2, rplusOk);
				n *= -1;
				break;
			case AOP_VMAX:
				n = gvmax_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxSuf->mxmDevice, mtxRet->mxmDevice, primesz,
					idxOrigin, widthPer, aopTrk2, rplusOk);
				break;
			}
			chron_end(lap, mtxPref, mtxRet, "one", "gpu", n);
			if(n > 0) {
				CopyDeviceToHost2(mtxRet, idxOrigin * widthPer, n, 1);
			} else srGate->checkSide(true);
			rutra->puthGpu(execGpu, n < 0 ? n * -1 : n);
		} else {
			switch(aopTrk) {
			case AOP_ACTF:
				n = cactf_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf ? mtxSuf->mxmHost : nullx, mtxRet->mxmHost, primesz,
					idxOrigin, widthPer, aopTrk2, rplusOk);
				break;
			case AOP_ACTF2:
			{
				ovar = (OneVar *)tcxtrTrk->mCxtHost;
				Matrix<DT> *m1 = *(Matrix<DT> **)ovar->idxOne;
				Matrix<DT> *m2 = *(Matrix<DT> **)&ovar->idxOne[2];
				n = cactf2_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf->mxmHost, mtxRet->mxmHost,
					m1->mxmHost, m2->mxmHost, primesz, idxOrigin, widthPer, aopTrk2, rplusOk);
				//m2도 갱신되므로 copy device memory해야하나 m2는 이 함수에서만 사용되므로 모드간 
				//전환되어 엑세스될 일 없으므로 하지 않는다.
			}
			break;
			case AOP_EMBEDDING:
				n = cembedding_t<DT>(mtxPref->mxmHost, mtxSuf->mxmHost, mtxRet->mxmHost, primesz,
					idxOrigin, widthPer, SZ_MTX_LOW_FIRST(mtxPref), mtxSuf->mxType,
					aopTrk2 ? MTX_SIZE(mtxRet) : MTX_SIZE(mtxPref), aopTrk2);
				n *= -1;
				//srGate->checkSide(false);//역전파일만 pref를 기준으로 하여 순전파일때는 여기서 복사해도 되나 일관성을 위해 순전파도 후 복사한다.
				break;
			case AOP_SLICE:
				n = cslice_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxRet->mxmHost, primesz,
					idxOrigin, widthPer, aopTrk2, rplusOk);
				n *= -1;
				//srGate->checkSide(false);
				break;
			case AOP_ONEHOT:
				n = conehot_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxRet->mxmHost, primesz,
					idxOrigin, widthPer);
				n *= -1;
				//srGate->checkSide(false);//pref를 기준으로하므로 후 복사한다.
				break;
			case AOP_ARGMAX:
				n = cargmax_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxRet->mxmHost, primesz,
					idxOrigin, widthPer);
				break;
			case AOP_EQUAL:
				n = cequal_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf ? mtxSuf->mxmHost : nullx, mtxRet->mxmHost, primesz,
					idxOrigin, widthPer);
				break;
			case AOP_TYPE1:
				n = ctype1_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf ? mtxSuf->mxmHost : nullx, mtxRet->mxmHost, primesz,
					idxOrigin, widthPer, aopTrk2);
				break;
			case AOP_RANDOM:
				if(mtxPref->mxType < memput::mp::tlong) {
					n = crandom_t<DT, floatt>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf ? mtxSuf->mxmHost : nullx,
						mtxRet ? mtxRet->mxmHost : nullx, primesz, idxOrigin, widthPer, aopTrk2, mtxPref->mxType, tcxtrTrk->tcxttrc->rseed);
				} else {
					n = crandom_t<DT, doublet>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf ? mtxSuf->mxmHost : nullx,
						mtxRet ? mtxRet->mxmHost : nullx, primesz, idxOrigin, widthPer, aopTrk2, mtxPref->mxType, tcxtrTrk->tcxttrc->rseed);
				}
				break;
			case AOP_LAYNOR:
				p = (DT **)tcxtrTrk->mCxtHost;
				n = clayer_norm_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxRet->mxmHost, *p,
					*(p + 1), *(p + 2), *(p + 3), *(p + 4), *(p + 5), *(p + 6), *(p + 7),
					*(p + 8), *(p + 9), *(p + 10), primesz, idxOrigin, widthPer, aopTrk2, rplusOk);
				n *= -1;
				//srGate->checkSide(false);
				break;
			case AOP_MATMUL:
				ovar = (OneVar *)tcxtrTrk->mCxtHost;
				n = matmul_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf->mxmHost, mtxRet->mxmHost,
					primesz, ovar->idxOne[0], ovar->idxOne[1], ovar->idxOne[2], ovar->idxOne[3], rplusOk,
					idxOrigin, widthPer);
				break;
			case AOP_BSUM:
				n = cbatch_sum_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxRet->mxmHost, primesz,
					idxOrigin, widthPer, aopTrk2, rplusOk);
				n *= -1;
				break;
			case AOP_VMAX:
				n = cvmax_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf->mxmHost, mtxRet->mxmHost, primesz,
					idxOrigin, widthPer, aopTrk2, rplusOk);
				break;
			}
			chron_end(lap, mtxPref, mtxRet, "one", "cpu", n);
			if(n > 0) {
				CopyHostToDevice2(mtxRet, idxOrigin * widthPer, n, 1);
			} else srGate->checkSide(false);
			rutra->puthGpu(nullx, n < 0 ? n * -1 : n);
		}
		srGate->srReturn();
	}
};
template<typename DT> Tracker *OneTrack<DT>::trkPool = nullx;

#define TWO_TYPE1	0
template<typename DT> class TwoTrack : public RootTrack {
public:
	static Tracker *trkPool;
	intt aopTrk, aopTrk2;
	Matrix<DT> *mtxPref, *mtxSuf, *mtxRet, *bmtxPref, *bmtxSuf;

	TwoTrack(void *tcxt, intt iori, intt width, CudaHandle *hgpu, RunTrack *rt) : RootTrack(tcxt, iori, width, hgpu, rt)
	{
		trkType = 12;
	}
	void mSetTrack(Matrix<DT> *p_mtx, Matrix<DT> *s_mtx, Matrix<DT> *r_mtx, Matrix<DT> *bp_mtx, Matrix<DT> *bs_mtx, intt aop, intt aop2)
	{
		mtxPref = p_mtx; mtxSuf = s_mtx; mtxRet = r_mtx; aopTrk = aop;
		bmtxPref = bp_mtx; bmtxSuf = bs_mtx;
		aopTrk2 = aop2;
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		intt n;
		chron_begin(lap, mtxPref);
		if(execGpu) {
			switch(aopTrk) {
			case TWO_TYPE1:
				n = gtwo_f(tcxtrTrk->mCxtDevice, mtxPref->mxmDevice, mtxSuf->mxmDevice, mtxRet->mxmDevice,
					bmtxPref ? bmtxPref->mxmDevice : nullx, bmtxSuf ? bmtxSuf->mxmDevice : nullx,
					MTX_SIZE(mtxPref), idxOrigin, widthPer, aopTrk2);
				break;
			}
			chron_end(lap, mtxPref, mtxRet, "two", "gpu", n);
			if(bmtxPref) {
				CopyDeviceToHost2(mtxPref, idxOrigin * widthPer, n, 1);
				CopyDeviceToHost2(mtxSuf, idxOrigin * widthPer, n, 1);
			} else CopyDeviceToHost2(mtxRet, idxOrigin * widthPer, n, 1);
			rutra->puthGpu(execGpu, n < 0 ? n * -1 : n);
		} else {
			switch(aopTrk) {
			case TWO_TYPE1:
				n = ctwo_t<DT>(tcxtrTrk->mCxtHost, mtxPref->mxmHost, mtxSuf->mxmHost, mtxRet->mxmHost,
					bmtxPref ? bmtxPref->mxmHost : nullx, bmtxSuf ? bmtxSuf->mxmHost : nullx,
					MTX_SIZE(mtxPref), idxOrigin, widthPer, aopTrk2);
				break;
			}
			chron_end(lap, mtxPref, mtxRet, "two", "cpu", n);
			if(bmtxPref) {
				CopyHostToDevice2(mtxPref, idxOrigin * widthPer, n, 1);
				CopyHostToDevice2(mtxSuf, idxOrigin * widthPer, n, 1);
			} else CopyHostToDevice2(mtxRet, idxOrigin * widthPer, n, 1);
			rutra->puthGpu(nullx, n < 0 ? n * -1 : n);
		}
		srGate->srReturn();
	}
};
template<typename DT> Tracker *TwoTrack<DT>::trkPool = nullx;

#define r_copy_val_type(pbuf, sval, tp, len) {\
	switch(tp) {\
	case tshort:\
		for(intt i = 0;i < len; i++) {\
			*(pbuf + i) = *(shortt *)sval;\
		}\
		break;\
	case tfloat:\
		for(intt i = 0;i < len; i++) {\
			*(pbuf + i) = *(floatt *)sval;\
		}\
		break;\
	case tint:\
		for(intt i = 0;i < len; i++) {\
			*(pbuf + i) = *(intt *)sval;\
		}\
		break;\
	case tlong:\
		for(intt i = 0;i < len; i++) {\
			*(pbuf + i) = *(longt *)sval;\
		}\
		break;\
	case tdouble:\
		for(intt i = 0;i < len; i++) {\
			*(pbuf + i) = *(doublet *)sval;\
		}\
		break;\
	}\
}
#define copy_val_type(pbuf, sval, tp) {\
	switch(tp) {\
	case tshort:\
		*(shortt *)pbuf = *(shortt *)sval;\
		break;\
	case tfloat:\
		*(floatt *)pbuf = *(floatt *)sval;\
		break;\
	case tint:\
		*(intt *)pbuf = *(intt *)sval;\
		break;\
	case tlong:\
		*(longt *)pbuf = *(longt *)sval;\
		break;\
	case tdouble:\
		*(doublet *)pbuf = *(doublet *)sval;\
		break;\
	}\
}
#define copy_cval_type(pbuf, cval, tp) {\
	switch(tp) {\
	case tshort:\
		*(shortt *)pbuf = (shortt)cval;\
		break;\
	case tfloat:\
		*(floatt *)pbuf = (floatt)cval;\
		break;\
	case tint:\
		*(intt *)pbuf = (intt)cval;\
		break;\
	case tlong:\
		*(longt *)pbuf = (longt)cval;\
		break;\
	case tdouble:\
		*(doublet *)pbuf = (doublet)cval;\
		break;\
	}\
}
#define univ_val_type(dst, src, dtp, T, i) {\
	switch(dtp) {\
	case tshort:\
		*((shortt *)dst + i) = (shortt)*((T *)src + i);\
		break;\
	case tfloat:\
		*((floatt *)dst + i) = (floatt)*((T *)src + i);\
		/*printf("(%d: %f %f) ", i, *((floatt *)dst + i), (floatt)*((T *)src + i));*/\
		break;\
	case tint:\
		*((intt *)dst + i) = (intt)*((T *)src + i);\
		break;\
	case tlong:\
		*((longt *)dst + i) = (longt)*((T *)src + i);\
		break;\
	case tdouble:\
		*((doublet *)dst + i) = (doublet)*((T *)src + i);\
		break;\
	}\
}
#define adj_val_type(dst, src, dtp, stp, i) {\
	switch(stp) {\
	case tshort:\
		univ_val_type(dst, src, dtp, shortt, i);\
		break;\
	case tfloat:\
		univ_val_type(dst, src, dtp, floatt, i);\
		break;\
	case tint:\
		univ_val_type(dst, src, dtp, intt, i);\
		break;\
	case tlong:\
		univ_val_type(dst, src, dtp, longt, i);\
		break;\
	case tdouble:\
		univ_val_type(dst, src, dtp, doublet, i);\
		break;\
	}\
}
#define r_adj_val_type(dst, src, dtp, stp, len) {\
	switch(stp) {\
	case tshort:\
		for(intt i = 0;i < len; i++) {\
			univ_val_type(dst, src, dtp, shortt, i);\
		}\
		break;\
	case tfloat:\
		for(intt i = 0;i < len; i++) {\
			univ_val_type(dst, src, dtp, floatt, i);\
		}\
		break;\
	case tint:\
		for(intt i = 0;i < len; i++) {\
			univ_val_type(dst, src, dtp, intt, i);\
		}\
		break;\
	case tlong:\
		for(intt i = 0;i < len; i++) {\
			univ_val_type(dst, src, dtp, longt, i);\
		}\
		break;\
	case tdouble:\
		for(intt i = 0;i < len; i++) {\
			univ_val_type(dst, src, dtp, doublet, i);\
		}\
		break;\
	}\
}
#define FILL_UNIV_OP 0
#define ARANGE_UNIV_OP 1
#define RANDN_UNIV_OP 2
#define RANDU_UNIV_OP 3
#define DSTR_WRITE_OP 4
#define TYPED_WRITE_OP 5
#define COPY_H2D_OP	6
#define EXPONENT_UNIV_OP 7
#define EXPAND_UNIV_OP 8
#define MIN_MAX_V_OP 9
#define STD_NORMAL	10
#define X_NORMAL	11
#define SIN_POSITIONAL 12

#define SIZE_SHAPE(ndim, dims, sz) {\
	sz = 1;\
	for(intt i = 0;i < ndim; i++) sz *= *(dims + i);\
}
template<typename CT> Tracker *RunTrack::trkGet(TContext *tcxt, intx iori, intt width, sytet gpu_exec)
{
	Tracker *trk;

	LOCK_MUT_(mutrt);
	CudaHandle *hgpu = (gpu_exec ? gethGpu<CudaHandle>(gpu_exec, tcxt->ground_did) : nullx);
	if(CT::trkPool) {
		GET_LIST2(CT::trkPool, trk);//pool list
		trk->trkReset();
		((RootTrack *)trk)->mSetRtra(tcxt, iori, width, hgpu);
		UNLOCK_MUT_(mutrt);
		return trk;
	}
	trk = new(this)CT(tcxt, iori, width, hgpu, this);

	UNLOCK_MUT_(mutrt);
	trk->trkReset();

	return trk;
}
template<typename CT> Tracker *RunTrack::trkGet2(void)
{
	Tracker *trk;

	LOCK_MUT_(mutrt);
	if(CT::trkPool) {//mmm
		GET_LIST2(CT::trkPool, trk);//pool list
		trk->trkReset();
		UNLOCK_MUT_(mutrt);
		return trk;
	}
	trk = new(this)CT(this);

	UNLOCK_MUT_(mutrt);
	trk->trkReset();

	return trk;
}
template<typename CT> void RunTrack::trkPut(Tracker *trk, bool lock)
{
	if(lock) LOCK_MUT_(mutrt);
	CAT_LIST2(Tracker, CT::trkPool, trk);
	if(lock) UNLOCK_MUT_(mutrt);
}

class Matrixr : public Typer {
public:
	ubytet mxType;
	sytet lapType;
	sytet cpmhot;
	bool mxomut;
	bool realtied;
	//bool groundFocus;
	//bool settleShadow;//이 클래스에서 바로 쉐도우가 장착됐는지를 알수있게 한다.
	intt maxmSize;
	intt mxndim;
	intt *mxshape;
	intt mxranksz[MX_DIM];
	Trace *mxTcr;
	intt npAbrib;
	intt nbackw;
	//intt gidalcmat;
	intt didground, didFocus;
	ShadowCap *lshadow;
	hmutex mutmtx;
	RunTrack *mxRutra;
	Matrixr *ptrLeft, *ptrRight;

	virtual bool resizeing(intt ndim, intt *axid, intt gid) = 0;
	virtual void resetMemory(sytet gpu) = 0;
	virtual void amxm(intt ndim, intt *axid, sytet init, intt gid = -1, Matrixr *mast = nullptr) = 0;
	virtual void mtxrm(bool r_mut) = 0;
	virtual intt sizem(bool t_size = 0) = 0;
	virtual intt sizem2(intt n, bool bsize = 1) = 0;
	virtual void *begin_dp(void) = 0;
	virtual void *begin_p(intt off = 0) = 0;
	virtual void *begin_wp(intt off = 0) = 0;
	virtual void *end_p(void) = 0;
	virtual void *read_p(intt xid[], intt irank, intt *rsz, intt n = 0) = 0;
	virtual void write_p(intt xid[], void *dat, intt irank, intt wsz = 0) = 0;
	virtual void printo(sytet leaf_one = 0, sytet width = 1) = 0;
	virtual void iprinto(intt i) = 0;
	virtual intt uniform(TContext *tcxt, Univ *uni) = 0;
	virtual void *devmpoint(intt i) = 0;
	virtual void devmsetting(intt i, void *devm) = 0;
	virtual void devmcopy(void *tar, void *sor) = 0;
	virtual void groundSet(intt gid) = 0;
	virtual void shadowSet(void *dev_shadow, intt id_shadow) = 0;
	virtual void dockAlloc(void) = 0;
	virtual void copyHostToGround(void) = 0;
	virtual void arrangeDevice(bool ground_to_shadow, intt did_shadow, void *dev_shadow, bool data) = 0;
	virtual void inCopy(Matrixr *i_mtx, sytet gpu) = 0;
	virtual intt copyMemory(void *pm, sytet gpu, intt begin, intt size) = 0;
	virtual void mmean(intt sz) = 0;
	virtual void marith(TContext *tcxt, Matrixr *sar_mtx, Matrixr *rar_mtx, ArithVar *parv, ubytet tval, void *sval, void *rplus, sytet aop) = 0;
	virtual void mdot(TContext *tcxt, Matrixr *sdot_mtx, Matrixr *rdot_mtx, DotVar *pdotv, sytet trans_order, bool cdot1, void *rplus) = 0;
	virtual void msplit(TContext *tcxt, void *pload, intt nload, intt axis, bool stacking, bool bw, bool parity) = 0;
	virtual void mconcat(TContext *tcxt, void *pload, intt nload, intt axis, bool stacking, bool bw, bool parity) = 0;
	virtual void mtranspose(TContext *tcxt, void *ret_mtx, TransVar *ptsvar, bool bw) = 0;
	virtual void msoftmax(TContext *tcxt, Matrixr *rmtx, Matrixr *sum_mtx, Matrixr *max_mtx, Matrixr *buf_mtx, OprVar1 *poprv) = 0;
	virtual void msoftx_cross_e(TContext *tcxt, void *ret_mtx, void *tar_mtx) = 0;
	virtual void mmean_square_e(TContext *tcxt, Matrixr *tar_mtx, void *ret_mtx, bool mean) = 0;
	virtual void msum(TContext *tcxt, void *ret_mtx, void *cmul, bool mean) = 0;
	virtual void moptadm(TContext *tcxt, void *v_mtx, void *g_mtx, void *r_mtx, floatt beta1,
		floatt beta2, floatt lr, floatt e, intt dec) = 0;
	virtual void moptsgd(TContext *tcxt, void *r_mtx, floatt lr, intt dec) = 0;
	virtual void mone(TContext *tcxt, Matrixr *s_mtx, Matrixr *r_mtx, sytet isz, OneVar *pvar, intt aop, intt aop2, sytet pdiv, sytet rplus, intt feat_sz = 0) = 0;
	virtual void mtwo(TContext *tcxt, Matrixr *s_mtx, Matrixr *r_mtx, Matrixr *bp_mtx, Matrixr *bs_mtx, intt aop, intt aop2) = 0;
	virtual void mclip(TContext *tcxt, Matrixr *r_mtx, doublet low, doublet high) = 0;
	virtual void mlayer_norm(TContext *tcxt, Matrixr *r_mtx, Matrixr *md, Matrixr *mz, Matrixr *mv, Matrixr *mean,
		Matrixr *g_mz, Matrixr *var, Matrixr *tmp, Matrixr *ga, Matrixr *be, Matrixr *g_ga, Matrixr *g_be, bool bw) = 0;
	virtual void mdiag_mul(TContext *tcxt, Matrixr *s_mtx, Matrixr *r_mtx) = 0;
	virtual void mdiag_fill(TContext *tcxt, Matrixr *r_mtx) = 0;
	void mlisting(void)
	{
		mxTcr->listm(this);
	}
	ShadowCap *findsap(intt sz, intt gid)
	{
		for(ShadowCap *sap = lshadow; sap; sap = sap->ptrRight) {
			if(sap->didshadow == gid && sz <= sap->shadowsz) return sap;
		}
		return nullptr;
	}
	ShadowCap *shadowFree(intt gid = -1)
	{
		for(ShadowCap *sap = lshadow; sap; sap = sap->ptrRight) {
			if(gid >= 0 && gid != sap->didshadow) continue;
			CudaDevSet(sap->didshadow);//메모리 해제는 생성 디바이스에 포커스하여 수행한다. 바로 밑에서 
					//헐당할때 그래프 수행 쓰레드에서 주어진 장비에 포커스되므로 포커스 변경문제 없다.
			cudaError_t error = cudaFree(sap->devShadow);
			cudaerror(error, "shadow free error");
			cudaDeviceSynchronize();
			//printf("cuda shadow free %p\n", devShadow);
			sap->devShadow = nullptr;
			if(gid >= 0) return sap;
		}
		return nullptr;
	}
	ShadowCap *shadowAlloc(intt sz, intt gid)
	{
		ShadowCap *sap;
		//printf("shadow alloc %d\n", gid);
		if(cpmhot > 0) copyHostToGround();
		
		sap = findsap(sz, gid);
		if(sap) return sap;

		if(sz < 0) sz = sizem();
		sap = shadowFree(gid);
		if(sap == nullptr) {
			sap = (ShadowCap *)mxTcr->xalloc(sizeof(ShadowCap));
			APPEND_LIST(lshadow, sap);
		}
		sap->didshadow = gid;
		sap->shadowsz = sz;
		CudaDevSet(gid);//그래프 수행 쓰레드와 장비 공유된다.
		cudaError_t error = cudaMalloc((void**)&sap->devShadow, sz);
		cudaerror(error, "shadow malloc error");
		cudaMemset(sap->devShadow, 0x00, sz);
		cudaDeviceSynchronize();
		//printf("cuda shadow alloc %p\n", devShadow);
		//settleShadow = 1;
		return sap;
	}
	void shape(void)
	{
		intt i = 0;

		printf("(");
		for(;i < mxndim - 1; i++) {
			printf("%d, ", mxshape[i]);
		}
		printf("%d)\n", mxshape[i]);
	}
	static intt make_rank_sz(intt ndim, intt *axid, intt *ranksz)
	{
		intt i, mxsz = 1;
		for(i = ndim - 1;i >= 0; i--) {
			if(*(axid + i) == 0) throwFault(-1, "not positive dimension: %d %d\n", *(axid + i), i);
			mxsz *= *(axid + i);
			*(ranksz + i) = (*(axid + i) == 1 ? mxsz * -1 : mxsz);
		}
		return mxsz;
	}
	static void shrink_dims(intt ndim, intt *axid, intt *rxid, intt axis)
	{
		intt i = 0, j = 0;
		for(;i < ndim; i++) {
			if(i != axis) *(rxid + j++) = *(axid + i);
		}
	}
	void fill(void *cv, sytet stp)
	{
		Univ uv(FILL_UNIV_OP, mxType);
		adj_val_type(&uv.cvuni, cv, mxType, stp, 0);
		uniform(nullx, &uv);
	}
	void expofill(intt exp_c)
	{
		Univ uv(EXPONENT_UNIV_OP, 0);
		uv.cvuni = exp_c;
		uniform(nullx, &uv);
	}
	void expandelen(Matrixr *src, intt n, intt axis)
	{
		Univ uv(EXPAND_UNIV_OP, 0);
		uv.cvuni = (longt)src;
		*(intt *)&uv.cvuni2 = n;
		*((intt *)&uv.cvuni2 + 1) = axis;
		uniform(nullx, &uv);
	}
	void msetArithv(Matrixr *sufmat, Matrixr *retmat, ArithVar *arv, ubytet t_scalv)
	{
		if(arv == nullx || arv->paintVar) return;
		arv->paintVar = 1;
		arv->bwGetOri = 0;
		arv->narPre = arv->narSuf = 0;
		if(t_scalv) {
			if(sufmat) {
				if(sufmat->mxType != t_scalv) throwFault(-1, "type fault\n");
				arv->narSuf = sufmat->mxndim;
				arv->zarSuf = MTX_SIZE(sufmat);
				memcpy(arv->arRankSuf, sufmat->mxranksz, arv->narSuf * sizeof(intt));
			} else {
				if(mxType != t_scalv) throwFault(-1, "type fault\n");
				arv->narPre = mxndim;
				arv->zarPre = MTX_SIZE(this);
				memcpy(arv->arRankPre, mxranksz, arv->narPre * sizeof(intt));
			}
		} else {
			arv->narPre = mxndim;
			arv->zarPre = MTX_SIZE(this);
			memcpy(arv->arRankPre, mxranksz, arv->narPre * sizeof(intt));
			arv->narSuf = sufmat->mxndim;
			arv->zarSuf = MTX_SIZE(sufmat);
			memcpy(arv->arRankSuf, sufmat->mxranksz, arv->narSuf * sizeof(intt));
		}
		arv->narMast = retmat->mxndim;
		arv->zarOut = MTX_SIZE(retmat);
		memcpy(arv->arRankMast, retmat->mxranksz, arv->narMast * sizeof(intt));
		if(arv->narPre == 0) arv->tpArith = AR_T_BROLC;
		else if(arv->narSuf == 0) arv->tpArith = AR_T_BRORC;
		else arv->tpArith = AR_T_O2O;
	}
};