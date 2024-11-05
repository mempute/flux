#pragma once

#include "rsc.h"
#include <random>

#ifdef __cplusplus
extern "C" {
#endif
	extern intt idx2offset(intt ndim, intt *srank, intt *idx);
	extern intt sparse_idx2offset(intt ndim, intt *srank, intt *idx, intt *axis);
	extern void offset2idx(intt ndim, intt *srank, intt off, intt *idx);
	extern void offset2idx2(intt out_axis[], intt n_preout_axis, intt rdim, intt rrank[], intt off, intt pidx[], intt sidx[]);
	extern void mulf(intt roff, intt n, floatt *m_pari, floatt *m_sari, floatt *m_rari, floatt rplus);
#ifdef __cplusplus
}
#endif

#define SM_SIZE	8192//32768
#define MRANK_SIZE(m_rank, ik) (*(m_rank + ik) < 0 ? *(m_rank + ik) * -1 : *(m_rank + ik))
#define MRANK_SIZE2(m_rank, ik) *(m_rank + ik)

#ifdef __cplusplus
#define PUT_IN_REGISTER
#else
#define PUT_IN_REGISTER register
#endif

template<typename T>
void gemm_nn(int M, int N, int K, T ALPHA,
	T *A, int lda,
	T *B, int ldb,
	T *C, int ldc)
{
	int i, j, k;
	for(i = 0; i < M; ++i) {
		for(k = 0; k < K; ++k) {
			PUT_IN_REGISTER T A_PART = ALPHA * A[i * lda + k];
			for(j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART * B[k*ldb + j];
			}
		}
	}
}
template<typename T>
void gemm_nn_fast(int M, int N, int K, T ALPHA,
	T *A, int lda,
	T *B, int ldb,
	T *C, int ldc)
{
	int i, j, k;
#pragma omp parallel for
	for(i = 0; i < M; ++i) {
		for(k = 0; k < K; ++k) {
			PUT_IN_REGISTER T A_PART = ALPHA * A[i*lda + k];
			for(j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART * B[k*ldb + j];
			}
		}
	}
}
template<typename T>
void gemm_nt(int M, int N, int K, T ALPHA,
	T *A, int lda,
	T *B, int ldb,
	T *C, int ldc)
{
	int i, j, k;
	for(i = 0; i < M; ++i) {
		for(j = 0; j < N; ++j) {
			PUT_IN_REGISTER T sum = 0;
			for(k = 0; k < K; ++k) {
				sum += ALPHA * A[i*lda + k] * B[j*ldb + k];
			}
			C[i*ldc + j] += sum;
		}
	}
}
template<typename T>
void gemm_tn(int M, int N, int K, T ALPHA,
	T *A, int lda,
	T *B, int ldb,
	T *C, int ldc)
{
	int i, j, k;
	for(i = 0; i < M; ++i) {
		for(k = 0; k < K; ++k) {
			PUT_IN_REGISTER T A_PART = ALPHA * A[k * lda + i];
			for(j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART * B[k*ldb + j];
			}
		}
	}
}
template<typename T>
void gemm_tt(int M, int N, int K, T ALPHA,
	T *A, int lda,
	T *B, int ldb,
	T *C, int ldc)
{
	int i, j, k;
	for(i = 0; i < M; ++i) {
		for(j = 0; j < N; ++j) {
			PUT_IN_REGISTER T sum = 0;
			for(k = 0; k < K; ++k) {
				sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
			}
			C[i*ldc + j] += sum;
		}
	}
}
//int is_avx() {
//	return 0;
//}
//int is_fma_avx2() {
//	return 0;
//}
template<typename T>
int gemm_cpu(int TA, int TB, int M, int N, int K, T ALPHA,
	T *A, int lda,
	T *B, int ldb,
	T BETA,
	T *C, int ldc)
{
	//printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
	if(BETA == 0) {
		int i, j;
		for(i = 0; i < M; ++i) {
			for(j = 0; j < N; ++j) {
				C[i*ldc + j] = 0;
			}
		}
	} else if(BETA != 1) {
		int i, j;
		for(i = 0; i < M; ++i) {
			for(j = 0; j < N; ++j) {
				C[i*ldc + j] *= BETA;
			}
		}
	}

	//is_avx();   // initialize static variable
	if(!TA && !TB) {//if(is_fma_avx2() && !TA && !TB) {
		gemm_nn_fast(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	} else {
		int t;
#pragma omp parallel for
		for(t = 0; t < M; ++t) {
			if(!TA && !TB)
				gemm_nn(1, N, K, ALPHA, A + t * lda, lda, B, ldb, C + t * ldc, ldc);
			else if(TA && !TB)
				gemm_tn(1, N, K, ALPHA, A + t, lda, B, ldb, C + t * ldc, ldc);
			else if(!TA && TB)
				gemm_nt(1, N, K, ALPHA, A + t * lda, lda, B, ldb, C + t * ldc, ldc);
			else
				gemm_tt(1, N, K, ALPHA, A + t, lda, B, ldb, C + t * ldc, ldc);
		}
	}
	return CUBLAS_STATUS_SUCCESS;
}
template<typename T>
void mdot2d(cublasHandle_t hcuda, int TA, int TB, int M, int N, int K, T ALPHA,
	T *pda, int lda, T *pdb, int ldb, T BETA, T *cdb, int ldc, sytet db, bool cpu)
{
	int m = (TA ? K : M);
	int n = (TB ? ldc : N);
	int k = (TA ? M : K);
	if(cpu) {
		gemm_cpu((TA ? CUBLAS_OP_T : CUBLAS_OP_N), (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
			m, n, k, ALPHA, pda, lda, pdb, ldb, BETA, cdb, ldc);
	} else {
		cublasStatus_t stat;
		if(db) stat = cublasDgemm(hcuda, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
			n, m, k, (doublet *)&ALPHA, (doublet *)pdb, ldb, (doublet *)pda, lda, (doublet *)&BETA, (doublet *)cdb, ldc);
		else stat = cublasSgemm(hcuda, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
			n, m, k, (floatt *)&ALPHA, (floatt *)pdb, ldb, (floatt *)pda, lda, (floatt *)&BETA, (floatt *)cdb, ldc);
		if(stat != 0) throwFault(stat, "cublas fail\n");
		if(stat != CUBLAS_STATUS_SUCCESS)
			cout << "cannot cublasSgemm dot" << endl;
		cudaDeviceSynchronize();
	}
}
#define SET_LINK_VAR(plv, mptr) plv->pheadVar = mptr;
#define P_LINK_VAR(dtp, plv, varoff) (dtp *)((bytet *)plv->pheadVar + varoff)
#define P_LINK_VAR2(dtp, mptr, varoff) (dtp *)((bytet *)mptr + varoff)
typedef struct {
	intt szRankPrimary, sdimCat, sbaseCat;
	intt mptrDevSecondary, mptrHostSecondary;
} ConcatVar;
template<typename T>
intt csplit_t(void *pcxt, T *m_split, intt pdim, intt sdim, intt rsize, intt idx_origin, intt idx_width, 
	intt nsplit, intt nstep, intt axis, bool bw)
{
	ConcatVar *ccv = (ConcatVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (rsize > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rsize);
	intt *prank;
	T *rsplit_mhost, *rsplit_mhosts;

	prank = P_LINK_VAR2(intt, pcxt, ccv->szRankPrimary);
	rsplit_mhosts = P_LINK_VAR2(T, pcxt, ccv->mptrHostSecondary);
	intt outer_sz = MRANK_SIZE(prank, axis);
	intt oi, si, soff, ns;

	if(nsplit > 0) {
		intt inner_sz = (axis == pdim - 1 ? 1 : MRANK_SIZE(prank, axis + 1)) * nstep;
		oi = roff / outer_sz;//split축 랭크 사이즈 단위로 각 분할 메모리에서 inner_sz단위로 적재됨
		soff = roff % outer_sz;//split축 랭크내의 분할 옵셋
		si = soff / inner_sz;//split축 분할 단위 이번 분할번째
		soff = oi * inner_sz + soff % inner_sz;//분할번째내의 옵셋
		ns = (oi + 1) * inner_sz;
		rsplit_mhost = *((T **)rsplit_mhosts + si);//이번 분할 메모리

		for(;roff < n;) {
			if(soff < ns) {//이번 분할내이면 이번 분할번재 적재 및 옵셋 증가
				if(bw) *(rsplit_mhost + soff) += *(m_split + roff);
				else *(rsplit_mhost + soff) = *(m_split + roff);
				//printf("[%d](%p) %d %d %f %f\n", si, rsplit_mhost, roff, soff, *(rsplit_mhost + soff), *(m_split + roff));
				soff++;
				roff++;
			} else {
				if(++si == nsplit) {
					si = 0;//split축 랭크 사이즈 단위를 넘었으면 분할 인덱스 초기화
					oi++;
					ns = (oi + 1) * inner_sz;
				}
				soff = oi * inner_sz;
				rsplit_mhost = *((T **)rsplit_mhosts + si);
			}
		}
	} else if(nstep) {//each map, cpu실행은 이 케이스로 실행되지 않으므로 필요없으나 그냥 둔다.
		intt *sbase = P_LINK_VAR2(intt, pcxt, ccv->sbaseCat);
		intt *sdim = P_LINK_VAR2(intt, pcxt, ccv->sdimCat);
		intt inner_sz = (axis == pdim - 1 ? 1 : MRANK_SIZE(prank, axis + 1));
		nsplit *= -1;
		oi = roff / outer_sz;
		soff = roff % outer_sz;//split축 랭크내의 분할 옵셋
		si = soff / inner_sz;//split축 분할 단위 이번 분할번째
		soff = oi * *(sdim + si) + (soff - *(sbase + si));//분할번째내의 옵셋
		ns = (oi + 1) * *(sdim + si);
		rsplit_mhost = *((T **)rsplit_mhosts + si);//이번 분할 메모리

		for(;roff < n;) {
			if(soff < ns) {//이번 분할내이면 이번 분할번재 적재 및 옵셋 증가
				if(bw) *(rsplit_mhost + soff) += *(m_split + roff);
				else *(rsplit_mhost + soff) = *(m_split + roff);
				soff++;
				roff++;
			} else {
				si += *(sdim + si) / inner_sz;
				if(si == nsplit) {
					si = 0;//split축 랭크 사이즈 단위를 넘었으면 분할 인덱스 초기화
					oi++;
				}
				if(*((T **)rsplit_mhosts + si) != rsplit_mhost) {
					soff = oi * *(sdim + si);//분할 매트릭스내에서 아웃터 oi번째의 시작 옵셋, 분할 매트릭스내이므로
					ns = (oi + 1) * *(sdim + si);//sdim단위로 균등하고 sdim단위의 oi가 하나 증가한번째가 경계가 된다.
					rsplit_mhost = *((T **)rsplit_mhosts + si);
				}
			}
		}
	} else {
		intt *sbase = P_LINK_VAR2(intt, pcxt, ccv->sbaseCat);
		intt *sdim = P_LINK_VAR2(intt, pcxt, ccv->sdimCat);
		intt inner_sz = (axis == pdim - 1 ? 1 : MRANK_SIZE(prank, axis + 1));
		nsplit *= -1;
		oi = roff / outer_sz;
		soff = roff % outer_sz;//split축 랭크내의 분할 옵셋
		for(si = 0;*(sbase + si) + *(sdim + si) <= soff; si++);//split축 분할 단위 이번 분할번째 find
		soff = oi * *(sdim + si) + (soff - *(sbase + si));//split축 분할 단위 이번 분할번째내의 옵셋
		ns = (oi + 1) * *(sdim + si);
		rsplit_mhost = *((T **)rsplit_mhosts + si);//이번 분할 메모리

		for(;roff < n;) {
			if(soff < ns) {//이번 분할내이면 이번 분할번재 적재 및 옵셋 증가
				if(bw) *(rsplit_mhost + soff) += *(m_split + roff);
				else *(rsplit_mhost + soff) = *(m_split + roff);
				soff++;
				roff++;
			} else {
				if(++si == nsplit) {
					si = 0;//split축 랭크 사이즈 단위를 넘었으면 분할 인덱스 초기화
					oi++;
				}
				soff = oi * *(sdim + si);//분할 매트릭스내에서 아웃터 oi번째의 시작 옵셋, 분할 매트릭스내이므로
				ns = (oi + 1) * *(sdim + si);//sdim단위로 균등하고 sdim단위의 oi가 하나 증가한번째가 경계가 된다.
				rsplit_mhost = *((T **)rsplit_mhosts + si);
			}
		}
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt cconcat_t(void *pcxt, T *m_rcat, intt pdim, intt sdim, intt rsize, intt idx_origin, intt idx_width,
	intt ncat, intt nstep, intt axis, bool bw)
{
	ConcatVar *ccv = (ConcatVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (rsize > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rsize);
	intt *prank;
	T *pcat_mhost, *pcat_mhosts;

	prank = P_LINK_VAR2(intt, pcxt, ccv->szRankPrimary);
	pcat_mhosts = P_LINK_VAR2(T, pcxt, ccv->mptrHostSecondary);
	intt outer_sz = MRANK_SIZE(prank, axis);//합쳐진 매트릭스에서 (단말에서)분할 축까지의 사이즈
	intt oi, si, soff, ns;

	if(ncat > 0) {
		intt inner_sz = (axis == pdim - 1 ? 1 : MRANK_SIZE(prank, axis + 1)) * nstep;
		oi = roff / outer_sz;
		soff = roff % outer_sz;//split축 랭크내의 분할 옵셋
		si = soff / inner_sz;//split축 분할 단위 이번 분할번째
		soff = oi * inner_sz + soff % inner_sz;//분할번째내의 옵셋
		ns = (oi + 1) * inner_sz;
		pcat_mhost = *((T **)pcat_mhosts + si);//이번 분할 메모리

		for(;roff < n;) {
			if(soff < ns) {//이번 분할내이면 이번 분할번재 적재 및 옵셋 증가
				if(bw) *(m_rcat + roff) += *(pcat_mhost + soff);
				else *(m_rcat + roff) = *(pcat_mhost + soff);
				soff++;
				roff++;
			} else {
				if(++si == ncat) {
					si = 0;//split축 랭크 사이즈 단위를 넘었으면 분할 인덱스 초기화
					oi++;
					ns = (oi + 1) * inner_sz;
				}
				soff = oi * inner_sz;
				pcat_mhost = *((T **)pcat_mhosts + si);//
			}
			//printf("[%d](%p) %d %d %d %f %f\n", i, pcat_mhost, sdim, roff, cat_off, *(m_rcat + roff), *(pcat_mhost + cat_off));
		}
	} else if(nstep) {//each map, cpu실행은 한번만 분할옵셋에 의해 분할번째를 찾고 이후는 반복이므로 분할차원의 매 원소별로 옵셋을  
		intt *sbase = P_LINK_VAR2(intt, pcxt, ccv->sbaseCat);//기록하여 속도를 높히는 실익이 크지 않으므로 이 케이스로 실행되지 
		intt *sdim = P_LINK_VAR2(intt, pcxt, ccv->sdimCat);	//않으므로 필요없으나 gpu와 매챙 차원에서 그냥 둔다.
		intt inner_sz = (axis == pdim - 1 ? 1 : MRANK_SIZE(prank, axis + 1));
		ncat *= -1;
		oi = roff / outer_sz;
		soff = roff % outer_sz;//split축 랭크내의 분할 옵셋
		si = soff / inner_sz;//split축 분할 단위 이번 분할번째
		soff = oi * *(sdim + si) + (soff - *(sbase + si));//분할번째내의 옵셋
		ns = (oi + 1) * *(sdim + si);
		pcat_mhost = *((T **)pcat_mhosts + si);//이번 분할 메모리

		for(;roff < n;) {
			if(soff < ns) {//이번 분할내이면 이번 분할번재 적재 및 옵셋 증가
				if(bw) *(m_rcat + roff) += *(pcat_mhost + soff);
				else *(m_rcat + roff) = *(pcat_mhost + soff);
				//printf("[%d][%d](%p) %d %d %f %f\n", *(sdim + si), *(sbase + si), pcat_mhost, roff, soff, *(m_rcat + roff), *(pcat_mhost + soff));
				soff++;
				roff++;
			} else {
				si += *(sdim + si) / inner_sz;
				if(si == ncat) {
					si = 0;//split축 랭크 사이즈 단위를 넘었으면 분할 인덱스 초기화
					oi++;
				}
				if(*((T **)pcat_mhosts + si) != pcat_mhost) {
					soff = oi * *(sdim + si);//분할 매트릭스내에서 아웃터 oi번째의 시작 옵셋, 분할 매트릭스내이므로
					ns = (oi + 1) * *(sdim + si);//sdim단위로 균등하고 sdim단위의 oi가 하나 증가한번째가 경계가 된다.
					pcat_mhost = *((T **)pcat_mhosts + si);
				}
			}
		} 
	} else {
		intt *sbase = P_LINK_VAR2(intt, pcxt, ccv->sbaseCat);
		intt *sdim = P_LINK_VAR2(intt, pcxt, ccv->sdimCat);
		intt inner_sz = (axis == pdim - 1 ? 1 : MRANK_SIZE(prank, axis + 1));
		ncat *= -1;
		oi = roff / outer_sz;
		soff = roff % outer_sz;//split축 랭크내의 분할 옵셋
		for(si = 0;*(sbase + si) + *(sdim + si) <= soff; si++);//split축 분할 단위 이번 분할번째 find
		soff = oi * *(sdim + si) + (soff - *(sbase + si));//split축 분할 단위 이번 분할번째내의 옵셋, 전자는 분할 매트릭스내의 옵셋, 후자는 통합 매트릭스에서 분할축내의 이번 분할매트릭스의 옵셋
		ns = (oi + 1) * *(sdim + si);
		pcat_mhost = *((T **)pcat_mhosts + si);//이번 분할 메모리

		for(;roff < n;) {
			if(soff < ns) {//이번 분할내이면 이번 분할번재 적재 및 옵셋 증가
				if(bw) *(m_rcat + roff) += *(pcat_mhost + soff);
				else *(m_rcat + roff) = *(pcat_mhost + soff);
				//printf("[%d][%d](%p) %d %d %f %f\n", *(sdim + si), *(sbase + si), pcat_mhost, roff, soff, *(m_rcat + roff), *(pcat_mhost + soff));
				soff++;
				roff++;
			} else {
				if(++si == ncat) {
					si = 0;//split축 랭크 사이즈 단위를 넘었으면 분할 인덱스 초기화
					oi++;
				}
				soff = oi * *(sdim + si);//분할 매트릭스내에서 아웃터 oi번째의 시작 옵셋, 분할 매트릭스내이므로
				ns = (oi + 1) * *(sdim + si);//sdim단위로 균등하고 sdim단위의 oi가 하나 증가한번째가 경계가 된다.
				pcat_mhost = *((T **)pcat_mhosts + si);
			}
		}
	}
	return n - idx_origin * idx_width;
}
#define _offset2idx(ndim, srank, _off, idx) {\
	intt j = 0, __off = _off;\
	for(intt i = 1;i < ndim; i++, j++) {\
		if(*(srank + j) < 0) *(idx + j) = 0;\
		else {\
			*(idx + j) = __off / MRANK_SIZE(srank, i);\
			__off %= MRANK_SIZE(srank, i);\
		}\
	}\
	*(idx + j) = __off;\
}
#define _idx2offset(ndim, srank, idx, __off) {\
	intt j = 0;\
	__off = 0;\
	for(intt i = 1;i < ndim; i++) {/*[2,1,1,3]일 경우 랭크는 [6,-3.-3, 3]이고 랭크가 1인 것이 두개 차원 연속되 3개 옵셋이 중복되거*/\
		__off += MRANK_SIZE(srank, i) * *(idx + j++);/*계산될것 같으나 idx가 [m,0,0,n]과 같이 랭크가 1인 차원은 idx가 0일 수밖에 */\
	}											/*없으므로 랭크가 1인 차원은 0이 곱해져 off가 위예에서 3개씩 중복되게 증가되지 않는다.*/\
	__off += *(idx + j);\
}
#define _lead_offset2idx(nbro, cdim, ndim, srank, _off, cidx, end_check) {\
	if(nbro) {\
		intt _i = 0;\
		for(;_i < cdim - ndim; _i++) cidx[_i] = 0;\
		_offset2idx(ndim, srank, _off, &cidx[_i]);\
	} else cidx[0] = _off;\
	end_check = 0;/*bro_offset의 ㄱ)에서 종료체크로 사용*/\
}
#define _bro_offset(nbro, bro_dim, bro_idx, cdim, crank, cidx, _off, end_check) {\
	intt i = nbro - 1;\
	if(end_check) break;/*ㄱ.*/\
	if(nbro) {\
		_idx2offset(cdim, crank, cidx, _off);\
		for(;i >= 0; i--) {\
			if(++cidx[bro_idx[i]] == bro_dim[i]) cidx[bro_idx[i]] = 0;\
			else break;\
		}\
		if(i < 0) end_check = 1;\
	} else {\
		end_check = 1;\
		_off = cidx[0];\
	}\
}
#define _moff2soff(mdim, mrank, sdim, srank, moff, idx, _off) {\
	intt i = mdim - 1, j = sdim - 1;\
	_offset2idx(mdim, mrank, moff, idx);\
	for(;j >= 0; i--, j--) {\
		if(*(srank + j) < 0) *(idx + i) = 0;\
	}\
	intt *ip = idx + ++i;\
	_idx2offset(sdim, srank, ip, _off);\
}
#define BWDIV_PREF 1
#define BWDIV_SUFF 2
typedef struct {
	sytet bwGetOri;
	sytet bopAtrith;
	sytet paintVar;
	sytet tpArith;
	intt narPre, narSuf, narMast, narRet, narBro;
	intt zarPre, zarSuf, zarOut;
	intt arRankPre[MX_DIM], arRankSuf[MX_DIM], arRankMast[MX_DIM], arRankRet[MX_DIM];
	intt broDimRet[MX_DIM], broIdxRet[MX_DIM];
} ArithVar;

#define ABP_MINUS_PREF	4
#define ABP_MINUS_SUFF	5
#define ABP_DIV_PREF	6
#define ABP_DIV_SUFF	7
#define ABP_BWTEST		8
//original - 모든 경우가 포함된 복합 버전
template<typename T>
intt carith_t(void *pcxt, T *m_pari, T *m_sari, T *m_rari, intt r_size, intt idx_origin, intt idx_width, T sval, sytet aop, T rplus)
{//Mul - 빈차원을 그 아래 차원이 순회하며 채워주므로 이에 맞춰 좌표계산
	ArithVar *arv = (ArithVar *)pcxt;
	intt roff = idx_origin * idx_width, coff;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt *mrank = arv->arRankMast, *prank = arv->arRankPre, *srank = arv->arRankSuf, *rrank = arv->arRankRet;
	intt npre = arv->narPre, nsuf = arv->narSuf, nmast = arv->narMast, nret = arv->narRet;
	intt off, cidx[MX_DIM], tmp_idx[MX_DIM];
	T rval, *ppre, *psuf;
	bool end_check;

	if(arv->bwGetOri) {
		for(;roff < n; roff++) {
			_lead_offset2idx(arv->narBro, npre, nret, rrank, roff, cidx, end_check);
			for(;;) {//rrank의 한개 원소에 대하여 rrank의 차원이 1인 랭크들을 순차적으로 순회하여 브로드캐스팅된 것들의 pre와 suf 연산
				_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, coff, end_check);
				if(coff < 0) break;
				if(m_pari) {//역전파에서 pref는 순전파에서 리턴되는 브로드캐스트된 매트릭스로 위에서 mrank로 옵셋계산되므로
					ppre = m_pari + coff;//바로 사용.
					//printf("%d ", coff);
				} else ppre = &sval;
				if(m_sari) {
					_moff2soff(nmast, mrank, nsuf, srank, coff, tmp_idx, off);
					psuf = m_sari + off;
					//printf("%d\n", off);
				} else psuf = &sval;
				switch(aop) {
				case AOP_MUL:
					rval = *ppre * *psuf;
					break;
				case AOP_PLUS:
					rval = *ppre;
					break;
				case AOP_DIV:
					break;
				case AOP_MINUS:
					break;
				case ABP_MINUS_PREF:
					rval = *ppre;
					break;
				case ABP_MINUS_SUFF:
					rval = *ppre * -1;
					break;
				case ABP_DIV_PREF:
					rval = *ppre * (1 / *psuf);
					break;
				case ABP_DIV_SUFF:
					rval = *ppre * (1 / (*psuf * *psuf) * -1);
					break;
				case ABP_BWTEST:
					rval = *ppre / *psuf;
					break;
				}
				if(aop == ABP_BWTEST) {
					if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
					else if(*(m_rari + roff) != rval) throwFault(-1, "xxx");
				} else {
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
			}
		}
	} else {
		for(;roff < n; roff++) {
			if(m_pari) {
				_moff2soff(nmast, mrank, npre, prank, roff, tmp_idx, off);
				ppre = m_pari + off;
				//printf("%d ", off);
			} else ppre = &sval;
			if(m_sari) {
				_moff2soff(nmast, mrank, nsuf, srank, roff, tmp_idx, off);
				psuf = m_sari + off;
				//printf("%d\n", off);
			} else psuf = &sval;
			switch(aop) {
			case AOP_MUL:
				rval = *ppre * *psuf;
				break;
			case AOP_PLUS:
				rval = *ppre + *psuf;
				break;
			case AOP_DIV:
				rval = *ppre / *psuf;
				break;
			case AOP_MINUS:
				rval = *ppre - *psuf;
				break;
			}
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
		}
	}
	return n - idx_origin * idx_width;
}
#define AR_T_O2O	0
#define AR_T_BROLC	1
#define AR_T_BRORC	2
#define AR_T_BRO	3
#define AR_T_ONEBRO	4
template<typename T>
intt carith_t1(void *pcxt, T *m_pari, T *m_sari, T *m_rari, intt r_size, intt idx_origin, intt idx_width, sytet aop, T rplus)
{//브로드 케스트가 없는 양측 매트릭스 일대일 대응 
	ArithVar *arv = (ArithVar *)pcxt;
	intt roff = idx_origin * idx_width, coff;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	T rval;//pre나 suf가 ret와 겹치면 pre나 suf가 계산되기위해 참조되기 전에 리셋되므로 먼저 게산하고 보관하기위해 
	if(arv->bwGetOri) {
		switch(aop) {
		case AOP_MUL:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) * *(m_sari + roff);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case AOP_PLUS:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case AOP_DIV:
			break;
		case AOP_MINUS:
			break;
		case ABP_MINUS_PREF:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_MINUS_SUFF:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) * -1;
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_DIV_PREF:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) * (1 / *(m_sari + roff));
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_DIV_SUFF:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) * (1 / (*(m_sari + roff) * *(m_sari + roff)) * -1);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_BWTEST:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) / *(m_sari + roff);
				if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
				else if(*(m_rari + roff) != rval) throwFault(-1, "xxx");
			}
			break;
		}
	} else {
		switch(aop) {
		case AOP_MUL:
			//mulf(n, roff, (floatt *)m_pari, (floatt *)m_sari, (floatt *)m_rari, (floatt)rplus);
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) * *(m_sari + roff);
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_PLUS:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) + *(m_sari + roff);
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_DIV:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) / *(m_sari + roff);
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_MINUS:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) - *(m_sari + roff);
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		}
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt carith_t2(void *pcxt, T *m_pari, T *m_sari, T *m_rari, intt r_size, intt idx_origin, intt idx_width, sytet aop, T rplus)
{//어느 한쪽이 작은 브로드 케스트(디멘젼이 1인 차원이 없는)
	ArithVar *arv = (ArithVar *)pcxt;
	intt roff = idx_origin * idx_width, coff;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt osz = arv->zarOut, ssz = arv->zarSuf, psz = arv->zarPre;
	T rval;
	if(arv->bwGetOri) {
		if(osz < psz) {//pre사이즈가 out보다 큰 경우, pre는 순전파에서 리턴 매트릭스이므로 pre가 더 적은 경우는 없음.
			switch(aop) {
			case AOP_MUL:
				for(;roff < n; roff++) {
					for(coff = roff;coff < psz; coff += osz) {
						rval = *(m_pari + coff) * *(m_sari + coff % ssz);
						if(rplus) *(m_rari + roff) += rval;
						else *(m_rari + roff) = rval;
					}
				}
				break;
			case AOP_PLUS:
				for(;roff < n; roff++) {
					for(coff = roff;coff < psz; coff += osz) {
						rval = *(m_pari + coff);
						if(rplus) *(m_rari + roff) += rval;
						else *(m_rari + roff) = rval;
					}
				}
				break;
			case AOP_DIV:
				break;
			case AOP_MINUS:
				break;
			case ABP_MINUS_PREF:
				for(;roff < n; roff++) {
					for(coff = roff;coff < psz; coff += osz) {
						rval = *(m_pari + coff);
						if(rplus) *(m_rari + roff) += rval;
						else *(m_rari + roff) = rval;
					}
				}
				break;
			case ABP_MINUS_SUFF:
				for(;roff < n; roff++) {
					for(coff = roff;coff < psz; coff += osz) {
						rval = *(m_pari + coff) * -1;
						if(rplus) *(m_rari + roff) += rval;
						else *(m_rari + roff) = rval;
					}
				}
				break;
			case ABP_DIV_PREF:
				for(;roff < n; roff++) {
					for(coff = roff;coff < psz; coff += osz) {
						rval = *(m_pari + coff) * (1 / *(m_sari + coff % ssz));
						if(rplus) *(m_rari + roff) += rval;
						else *(m_rari + roff) = rval;
					}
				}
				break;
			case ABP_DIV_SUFF:
				for(;roff < n; roff++) {
					for(coff = roff;coff < psz; coff += osz) {
						rval = *(m_pari + coff) * (1 / (*(m_sari + coff % ssz) * *(m_sari + coff % ssz)) * -1);
						if(rplus) *(m_rari + roff) += rval;
						else *(m_rari + roff) = rval;
					}
				}
				break;
			case ABP_BWTEST:
				for(;roff < n; roff++) {
					for(coff = roff;coff < psz; coff += osz) {
						rval = *(m_pari + coff) / *(m_sari + coff % ssz);
						if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
						else if(*(m_rari + roff) != rval) throwFault(-1, "xxx");
					}
				}
				break;
			}
		} else {//pre와 out의 사이즈가 같은 경우
			switch(aop) {
			case AOP_MUL:
				for(;roff < n; roff++) {
					rval = *(m_pari + roff) * *(m_sari + roff % ssz);
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
				break;
			case AOP_PLUS:
				for(;roff < n; roff++) {
					rval = *(m_pari + roff);
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
				break;
			case AOP_DIV:
				break;
			case AOP_MINUS:
				break;
			case ABP_MINUS_PREF:
				for(;roff < n; roff++) {
					rval = *(m_pari + roff);
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
				break;
			case ABP_MINUS_SUFF:
				for(;roff < n; roff++) {
					rval = *(m_pari + roff) * -1;
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
				break;
			case ABP_DIV_PREF:
				for(;roff < n; roff++) {
					rval = *(m_pari + roff) * (1 / *(m_sari + roff % ssz));
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
				break;
			case ABP_DIV_SUFF:
				for(;roff < n; roff++) {
					rval = *(m_pari + roff) * (1 / (*(m_sari + roff % ssz) * *(m_sari + roff % ssz)) * -1);
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
				break;
			case ABP_BWTEST:
				for(;roff < n; roff++) {
					rval = *(m_pari + roff) / *(m_sari + roff % ssz);
					if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
					else if(*(m_rari + roff) != rval) throwFault(-1, "xxx");
				}
				break;
			}
		}
	} else {
		switch(aop) {
		case AOP_MUL:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff % psz) * *(m_sari + roff % ssz);
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_PLUS:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff % psz) + *(m_sari + roff % ssz);
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_DIV:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff % psz) / *(m_sari + roff % ssz);
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_MINUS:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff % psz) - *(m_sari + roff % ssz);
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		}
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt carith_t2_lc(void *pcxt, T *m_sari, T *m_rari, intt r_size, intt idx_origin, intt idx_width, T sval, sytet aop, T rplus)
{//좌측이 상수인 브로드 케스트(디멘젼이 1인 차원이 없는)
	ArithVar *arv = (ArithVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	T rval;
	if(arv->bwGetOri) {//m_sari는 순전파때의 출력
		switch(aop) {
		case AOP_MUL:
			for(;roff < n; roff++) {
				rval = (sval * *(m_sari + roff));
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case AOP_PLUS:
			for(;roff < n; roff++) {
				rval = *(m_sari + roff);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case AOP_DIV:
			break;
		case AOP_MINUS:
			break;
		case ABP_MINUS_PREF://좌측이 상수이므로 pref는 없다.
			break;
		case ABP_MINUS_SUFF:
			for(;roff < n; roff++) {
				if(rplus) *(m_rari + roff) += *(m_sari + roff) * -1;
				else *(m_rari + roff) = *(m_sari + roff) * -1;
			}
			break;
		case ABP_DIV_PREF://좌측이 상수이므로 pref는 없다.
			break;
		case ABP_DIV_SUFF:
			for(;roff < n; roff++) {
				rval = *(m_sari + roff) * (1 / (*(m_rari + roff) * *(m_rari + roff)) * -1);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_BWTEST:
			for(;roff < n; roff++) {
				rval = sval / *(m_sari + roff);
				if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
				else if(*(m_rari + roff) != rval) throwFault(-1, "xxx");
			}
			break;
		}
	} else {
		switch(aop) {
		case AOP_MUL:
			for(;roff < n; roff++) {
				rval = (sval * *(m_sari + roff));
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_PLUS:
			for(;roff < n; roff++) {
				rval = (sval + *(m_sari + roff));
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_DIV:
			for(;roff < n; roff++) {
				rval = (sval / *(m_sari + roff));
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_MINUS:
			for(;roff < n; roff++) {
				rval = (sval - *(m_sari + roff));
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		}
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt carith_t2_rc(void *pcxt, T *m_pari, T *m_rari, intt r_size, intt idx_origin, intt idx_width, T sval, sytet aop, T rplus)
{//우측이 상수인 브로드 케스트(디멘젼이 1인 차원이 없는)
	ArithVar *arv = (ArithVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	T rval;
	if(arv->bwGetOri) {
		switch(aop) {
		case AOP_MUL:
			for(;roff < n; roff++) {
				rval = (*(m_pari + roff) * sval);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case AOP_PLUS:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case AOP_DIV:
			break;
		case AOP_MINUS:
			break;
		case ABP_MINUS_PREF:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_MINUS_SUFF://우측이 상수이므로 suff는 없다.
			//for(;roff < n; roff++) {
			//	rval = *(m_pari + roff) * -1;
			//	if(rplus) *(m_rari + roff) += rval;
			//	else *(m_rari + roff) = rval;
			//}
			break;
		case ABP_DIV_PREF:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) * (1 / sval);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_DIV_SUFF://우측이 상수이므로 suff는 없다.
			//for(;roff < n; roff++) {
			//	rval = *(m_pari + roff) * (1 / (sval * sval) * -1);
			//	if(rplus) *(m_rari + roff) += rval;
			//	else *(m_rari + roff) = rval;
			//}
			break;
		case ABP_BWTEST:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) / sval;
				if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
				else if(*(m_rari + roff) != rval) throwFault(-1, "xxx");
			}
			break;
		}
	} else {
		switch(aop) {
		case AOP_MUL:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) * sval;
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_PLUS:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) + sval;
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_DIV:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) / sval;
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_MINUS:
			for(;roff < n; roff++) {
				rval = *(m_pari + roff) - sval;
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		}
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt carith_t3(void *pcxt, T *m_pari, T *m_sari, T *m_rari, intt r_size, intt idx_origin, intt idx_width, sytet aop, T rplus)
{//디멘젼이 1인 차원을 포함하는 브로드 케스트, 어느 한쪽이 상수이면 상대측 디멘젼이 1인 차원이 의미가 없어지므로 타입 2번 케이스가 실행됨.
	ArithVar *arv = (ArithVar *)pcxt;
	intt roff = idx_origin * idx_width, poff, soff;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt *mrank = arv->arRankMast, *prank = arv->arRankPre, *srank = arv->arRankSuf, *rrank = arv->arRankRet;
	intt npre = arv->narPre, nsuf = arv->narSuf, nmast = arv->narMast, nret = arv->narRet;
	intt cidx[MX_DIM], tmp_idx[MX_DIM];
	T rval;
	bool end_check;

	if(arv->bwGetOri) {
		if(arv->zarOut < arv->zarPre) {//pre사이즈가 out보다 큰 경우, pre는 순전파에서 리턴 매트릭스이므로 pre가 더 적은 경우는 없음.
			switch(aop) {
			case AOP_MUL:
				for(;roff < n; roff++) {
					_lead_offset2idx(arv->narBro, npre, nret, rrank, roff, cidx, end_check);
					for(;;) {
						_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
						_moff2soff(nmast, mrank, nsuf, srank, poff, tmp_idx, soff);
						rval = *(m_pari + poff) * *(m_sari + soff);
						if(rplus) *(m_rari + roff) += rval;
						else *(m_rari + roff) = rval;
					}
				}
				break;
			case AOP_PLUS:
				for(;roff < n; roff++) {
					_lead_offset2idx(arv->narBro, npre, nret, rrank, roff, cidx, end_check);
					for(;;) {
						_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
						rval = *(m_pari + poff);
						if(rplus) *(m_rari + roff) += rval;
						else *(m_rari + roff) = rval;
					}
				}
				break;
			case AOP_DIV:
				break;
			case AOP_MINUS:
				break;
			case ABP_MINUS_PREF:
				for(;roff < n; roff++) {
					_lead_offset2idx(arv->narBro, npre, nret, rrank, roff, cidx, end_check);
					for(;;) {
						_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
						rval = *(m_pari + poff);
						if(rplus) *(m_rari + roff) += rval;
						else *(m_rari + roff) = rval;
					}
				}
				break;
			case ABP_MINUS_SUFF:
				for(;roff < n; roff++) {
					_lead_offset2idx(arv->narBro, npre, nret, rrank, roff, cidx, end_check);
					for(;;) {
						_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
						rval = *(m_pari + poff) * -1;
						if(rplus) *(m_rari + roff) += rval;
						else *(m_rari + roff) = rval;
					}
				}
				break;
			case ABP_DIV_PREF:
				for(;roff < n; roff++) {
					_lead_offset2idx(arv->narBro, npre, nret, rrank, roff, cidx, end_check);
					for(;;) {
						_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
						_moff2soff(nmast, mrank, nsuf, srank, poff, tmp_idx, soff);
						rval = *(m_pari + poff) * (1 / *(m_sari + soff));
						if(rplus) *(m_rari + roff) += rval;
						else *(m_rari + roff) = rval;
					}
				}
				break;
			case ABP_DIV_SUFF:
				for(;roff < n; roff++) {
					_lead_offset2idx(arv->narBro, npre, nret, rrank, roff, cidx, end_check);
					for(;;) {
						_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
						_moff2soff(nmast, mrank, nsuf, srank, poff, tmp_idx, soff);
						rval = *(m_pari + poff) * (1 / (*(m_sari + soff) * *(m_sari + soff)) * -1);
						if(rplus) *(m_rari + roff) += rval;
						else *(m_rari + roff) = rval;
					}
				}
				break;
			case ABP_BWTEST:
				for(;roff < n; roff++) {
					_lead_offset2idx(arv->narBro, npre, nret, rrank, roff, cidx, end_check);
					for(;;) {
						_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
						_moff2soff(nmast, mrank, nsuf, srank, poff, tmp_idx, soff);
						rval = *(m_pari + poff) / *(m_sari + soff);
						if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
						else if(*(m_rari + roff) != rval) throwFault(-1, "xxx");
					}
				}
				break;
			}
		} else {//pre와 out의 사이즈가 같은 경우
			switch(aop) {
			case AOP_MUL:
				for(;roff < n; roff++) {
					_moff2soff(nmast, mrank, nsuf, srank, roff, tmp_idx, soff);
					rval = *(m_pari + roff) * *(m_sari + soff);
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
				break;
			case AOP_PLUS:
				for(;roff < n; roff++) {
					rval = *(m_pari + roff);
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
				break;
			case AOP_DIV:
				break;
			case AOP_MINUS:
				break;
			case ABP_MINUS_PREF:
				for(;roff < n; roff++) {
					rval = *(m_pari + roff);
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
				break;
			case ABP_MINUS_SUFF:
				for(;roff < n; roff++) {
					rval = *(m_pari + roff) * -1;
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
				break;
			case ABP_DIV_PREF:
				for(;roff < n; roff++) {
					_moff2soff(nmast, mrank, nsuf, srank, roff, tmp_idx, soff);
					rval = *(m_pari + roff) * (1 / *(m_sari + soff));
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
				break;
			case ABP_DIV_SUFF:
				for(;roff < n; roff++) {
					_moff2soff(nmast, mrank, nsuf, srank, roff, tmp_idx, soff);
					rval = *(m_pari + roff) * (1 / (*(m_sari + soff) * *(m_sari + soff)) * -1);
					if(rplus) *(m_rari + roff) += rval;
					else *(m_rari + roff) = rval;
				}
				break;
			case ABP_BWTEST:
				for(;roff < n; roff++) {
					_moff2soff(nmast, mrank, nsuf, srank, roff, tmp_idx, soff);
					rval = *(m_pari + roff) / *(m_sari + soff);
					if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
					else if(*(m_rari + roff) != rval) throwFault(-1, "xxx");
				}
				break;
			}
		}
	} else {
		switch(aop) {
		case AOP_MUL:
			for(;roff < n; roff++) {
				_moff2soff(nmast, mrank, npre, prank, roff, tmp_idx, poff);
				_moff2soff(nmast, mrank, nsuf, srank, roff, tmp_idx, soff);
				rval = *(m_pari + poff) * *(m_sari + soff);
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_PLUS:
			for(;roff < n; roff++) {
				_moff2soff(nmast, mrank, npre, prank, roff, tmp_idx, poff);
				_moff2soff(nmast, mrank, nsuf, srank, roff, tmp_idx, soff);
				rval = *(m_pari + poff) + *(m_sari + soff);
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_DIV:
			for(;roff < n; roff++) {
				_moff2soff(nmast, mrank, npre, prank, roff, tmp_idx, poff);
				_moff2soff(nmast, mrank, nsuf, srank, roff, tmp_idx, soff);
				rval = *(m_pari + poff) / *(m_sari + soff);
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		case AOP_MINUS:
			for(;roff < n; roff++) {
				_moff2soff(nmast, mrank, npre, prank, roff, tmp_idx, poff);
				_moff2soff(nmast, mrank, nsuf, srank, roff, tmp_idx, soff);
				rval = *(m_pari + poff) - *(m_sari + soff);
				if(rplus) {
					*(m_rari + roff) *= rplus;
					*(m_rari + roff) += rval;
				} else *(m_rari + roff) = rval;
			}
			break;
		}
	}
	return n - idx_origin * idx_width;
}
#define _sparse_idx2offset(ndim, srank, idx, off) {\
	intt j = 0;\
	off = 0;\
	for(intt i = 1;i < ndim; i++) {\
		off += MRANK_SIZE(srank, i) * *(idx + j++);\
	}\
	off += *(idx + j);\
}
#define _jseq2offset(jseq, jdim, jrank, spr_jo, joff) {/*위 인덱스 설정 부분을 바로 joff증가로*/\
	intt j = 0, i = 1, off = jseq;\
	for(joff = 0;i < jdim; i++, j++) {\
		if(*(jrank + j) < 0);\
		else {\
			joff += spr_jo[j].rksz * (off / MRANK_SIZE(jrank, i));\
			off %= MRANK_SIZE(jrank, i);\
		}\
	}\
	joff += spr_jo[j].rksz * off;\
}
#define _jseq2offset2(jseq, jdim, jrank, spr_jo, idx, joff) {/*위와 동일, 인덱스 설정 병행*/\
	intt j = 0, i = 1, off = jseq;\
	for(joff = 0;i < jdim; i++, j++) {\
		if(*(jrank + j) < 0) *(idx + j) = 0;\
		else {\
			*(idx + j) = off / MRANK_SIZE(jrank, i);\
			joff += spr_jo[j].rksz * *(idx + j);\
			off %= MRANK_SIZE(jrank, i);\
		}\
	}\
	*(idx + j) = off;\
	joff += spr_jo[j].rksz * off;\
}
#define _offset2idx2(n_preout_axis, _nout, _orank, _off, pidx, sidx) {\
	intt j = 0, p = 0, s = 0, off = _off;\
	for(intt i = 1;i < _nout; i++, j++) {/*해당 인덱스는 하나 아래 랭크의 사이즈로 나눈 몫이므로 i는 1부터*/\
		if(j < n_preout_axis) {\
			if(_orank[j] < 0) pidx[p++] = 0;\
			else {\
				pidx[p++] = off / MRANK_SIZE(_orank, i);\
				off %= MRANK_SIZE(_orank, i);\
			}\
		} else {\
			if(_orank[j] < 0) sidx[s++] = 0;\
			else {\
				sidx[s++] = off / MRANK_SIZE(_orank, i);\
				off %= MRANK_SIZE(_orank, i);\
			}\
		}\
	}\
	sidx[s] = off;/*마지막 인덱스는 suf matrix의 것이 되고 남은 옵셋이 된다.*/\
}
typedef struct {
	bool rkPref;
	intt rksz, rktsz, rkdim;
} SparseRank;
typedef struct {
	SparseRank spr_jcache[MX_DIM], spr_jleaf[MX_DIM], spr_out[MX_DIM];
	intt jrank_cache[MX_DIM], jrank_leaf[MX_DIM], out_rank[MX_DIM];
	intt njo_cache, njo_leaf, cache_axis, roff_ori;
	intt oksz, jksz, nrecycc, ncycj, njoint, nout;
	bool okfit, jkfit;
} SharePoint;
//oksz은 블럭 사이즈가 되고 블럭내에서만 공유메모리가 공유되고 블럭사이즈가 쓰레드 걋수이고 쓰레드에 의해 동시에 공유메모리로 적재되므로 공유메모리 크기는 이 사이즈로 설정한다.
//jksz은 sz_smit가 블럭 사이즈보다 크면 oksz가 되고 아니면 sz_smit가 설정된다.
//jcyc - 초기값  0부터 시작, irecyc - 초기값 nrecyc. nrecycc, ncycj - 둘다 한번 더 반복할거면 1
#define cache_load(n, roff_ori, roff, okfit, jkfit, oksz, jksz, nrecycc, ncycj, njoint, irecyc, \
	jcyc, tid, bsz, inc, cache_axis, nout, out_rank, tmp_idx, spr_out, spr_jcache, spr_jleaf, njo_cache, \
	jrank_cache, njo_leaf, jrank_leaf, _m_pdot, m_pdot, _m_sdot, m_sdot, m_cache, m_leaf, leaf_idx, \
	cache_bank, lb) {\
	intt sz_smit = MRANK_SIZE(out_rank, cache_axis +1);/*ㄱ.pre나 suf중 맨끝에 위치한 것의 사이즈(연속하면 그중 최상위)*/\
	intt joff, jseq;\
	_offset2idx(nout, out_rank, roff, tmp_idx);\
	for(i = 0, m_pdot = _m_pdot, m_sdot = _m_sdot;i < nout; i++) {\
		if(spr_out[i].rkPref) m_pdot += (tmp_idx[i] * spr_out[i].rksz);\
		else m_sdot += (tmp_idx[i] * spr_out[i].rksz);\
	}/*m_pdot나 m_sdot 어느것이 됐던 위 ㄱ)의 조건을 만족하는 포인터가 획득된다.*/\
	if(spr_out[cache_axis].rkPref) {\
		m_cache = m_pdot;/*연속하는 맨 하위 차 상위 메모리(그 하위 한개마다 조인차원으로서*/\
		m_leaf = m_sdot;\
	} else {\
		m_cache = m_sdot;/*반복되야 하는 메모리중 최하위 메모리 포인터 획득)*/\
		m_leaf = m_pdot;\
	}\
	if(irecyc++ < nrecycc) {/*연속하는 맨 하위 차 상위 메모리의 조인 순열 재사용, 연속하는 최하단말사이즈(sz_smit)*/\
		if(okfit == 0 && irecyc == nrecycc) {/*가블럭사이즈(oksz)보다 커서 분할된경우로서 마지막 분할 구간(캐쉬 재사용 마지막 번째는)*/\
			inc = sz_smit % oksz;/*다음 옵셋 증가를 이번 남은 것을 처리한 것만큼만 증가 시키게 설정한다.*/\
			if(tid < inc) {\
				/*printf("******* %d %d %d %d \n", tid, ttid, sz_smit, sz_smit % oksz);*/\
			} else {/*남은 분할 갯수를 벗어난 tid들은 살행하지 않는다.*/\
				/*printf("++++++ %d %d %d %d \n", tid, ttid, sz_smit, sz_smit % oksz);*/\
				continue;\
			}\
		}\
	} else {/*연속하는 맨 하위 차 상위차원의 원소 한개에 대한 조인차원 원소들을 공유 메모리에 적재*/\
		/*printf("33 %d %d %d %d\n", jseq, jksz, njoint, sz_smit);*/\
		if(roff >= n) {/*출력 매트릭스를 모두 순회했으면*/\
			if(jcyc < ncycj) {/*조인 열이 분할되어 남아있으면 조인커널 사이즈 단위의*/\
				jcyc++;/*다음을 또 출력 매트릭스 전채에 대하여 순회 준비*/\
				roff = roff_ori;\
				if(okfit == 0) __syncthreads();/*ㄷ)설명과 동일*/\
				/*printf("------------\n");*/\
				goto lb;/*첫번째 수행은 roff이동하지 않기위해*/\
			} else {\
				break;/*더이상 조인열이 남아있지 않으면 전채 종료*/\
			}\
		}\
		if(inc != bsz) inc = bsz;/*원래 분할 사이즈로 원위치*/\
		irecyc = 0;\
		jseq = (tid % jksz) + jcyc * jksz;/*현 tid가 몇번째 조인 순열에 해당하는지 계산(njoint가 jksz보다 작아도 계산맞음)*/\
		if(okfit == 0) __syncthreads();/*ㄷ.sz_smit가 블럭 사이즈(oksz)를 넘어서 여러 블럭에 이어질 경우 이전 실행이 마지막으로서 블럭사이즈에 못미쳐 남는 쓰레드들이 실행되않았으므로 동기화*/\
		if(jseq < njoint) {/*ㄴ.총 조인갯수(njoint)가 조인커널(jksz)보다 적거나 njoint가 sz_smit보다 커서 분할되는 경우*/\
			/*분할 마지막은 jseq가 njoint이상 인 tid들은 총 조인사이즈를 벗어난 것들이므로 캐쉬에 적재않고 그 안쪽만 한다.*/\
			/*printf("44 %d %d\n", jseq, njoint);*/\
			_jseq2offset(jseq, njo_cache, jrank_cache, spr_jcache, joff);\
			/*printf("%d %d %d %d %f\n", jksz, oksz, joff, tid, *(m_cache + joff));*/\
			cache_bank[tid] = *(m_cache + joff);/*연속하는 맨 하위 사이즈단위(sz_smit) 옵셋의 tid의 조인커럴 배수 위치에 그 상위 조인 순열 적재*/\
		}\
		if(jcyc == ncycj && jkfit == 0) __syncthreads();/*조인순열의 사이즈가 sz_smit보다 적든지 분할된 마지막이 적던지간에 sz_smit에 맞지 않으면 ㄴ)이 스킵되므로 동기화수행, 분할 중간은 맞으므로 동기화 수행안함.*/\
	}\
	m_cache = &cache_bank[(tid / sz_smit) * sz_smit];/*연속하는 맨 하위 메모리(매트릭스)의 차 상위 메모리(매트릭스)의 캐쉬되어 재사용되는 조인시작 포인터 획득(매 조인 사이클 번째의), sz_smit가 블럭사이즈 이상일 경우 캐쉬메모리의 0번째*/\
	jseq = jcyc * jksz;\
	_jseq2offset2(jseq, njo_leaf, jrank_leaf, spr_jleaf, leaf_idx, joff);\
	m_leaf += joff;\
	/*printf("44 %d %d %d %d %f\n", tid, jseq, (tid / sz_smit) * sz_smit, joff, *m_leaf);*/\
}
typedef struct {
	sytet useCublas;
	sytet bwGetOri;
	sytet jdimEqual;
	sytet transOrder;//cublas에서만 사용.
	bool intervOut, retFirst, fitOutKernel, fitJoKernel;
	intt ndimPre, ndimSuf, ndimRet;
	intt nJointAxis;//joint axis들의 dim을 곱한 값
	intt njoPre, njoSuf, njoRet;//조인트되는 차원 갯수
	intt joAxisPre[MX_DIM], joAxisSuf[MX_DIM], joAxisRet[MX_DIM];//조인트되는 차원 인덱스, Ret의 경우 순전파때 조인된 축을 의미
	intt joRankPre[MX_DIM], joRankSuf[MX_DIM], joRankRet[MX_DIM];//양측에서 원소별로 곱해지는 차원 랭크 사이즈, dot 출력 모양에서 사라지게 된다.
	intt noutPre, noutSuf, noutRank, noutRet;//출력되는 차원 갯수
	intt outAxisPre[MX_DIM], outAxisSuf[MX_DIM], outAxisRet[MX_DIM];//출력되는 차원 인덱스, Ret의 경우 순전파때 출력 축을 의미
	intt outRankPre[MX_DIM], outRankSuf[MX_DIM];//조인트가 아닌 차원 랭크 사이즈, dot 출력 shape이 된다.
	intt szRankPre[MX_DIM], szRankSuf[MX_DIM], szRankRet[MX_DIM];//양측 매트릭스 각각의 dot되기 전의 완전한 차원 랭크 사이즈
	intt outAxis[MX_DIM], outRank[MX_DIM];
	SparseRank sprPreOut[MX_DIM], sprPreJo[MX_DIM], sprSufOut[MX_DIM], sprSufJo[MX_DIM];
	intt szSuf, szJoKernel, szOutKernel, nrecycCache, ncycJo, axisCache;
	intt shareUnit;//공유 메모리 다중 블럭에서만 사용.
	intt prem, sufn, joik, lda, ldb, ldc;//cublas에서만 사용.
	void *bwMxp, *bwMxs;
} DotVar;
template<typename T>
intt cdot_t(void *pcxt, T *m_pdot, T *m_sdot, T *m_rdot, intt rdot_size, intt idx_origin, intt idx_width, floatt rplus)
{
	DotVar *dotv = (DotVar *)pcxt;
	intt roff = idx_origin * idx_width;
	const intt n = (rdot_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rdot_size);
	intt *out_rank = dotv->outRank, nout = dotv->noutRank;
	intt njo_pre = dotv->njoPre, njo_suf = dotv->njoSuf;
	const bool bw_get_ori = dotv->bwGetOri, jdim_equal = dotv->jdimEqual;
	SparseRank *spr_out = dotv->sprPreOut, *spr_pre_jo = dotv->sprPreJo, *spr_suf_jo = dotv->sprSufJo;
	intt pj_idx[MX_DIM], sj_idx[MX_DIM], ret_idx[MX_DIM], i;
	T sum;
	//현 roff는 ret매트릭스가 기준인 옵셋이고 이것을 이번 매트릭스 곱의 양측 out axis rank기준으로 변환한다.
	_offset2idx(nout, out_rank, roff, ret_idx);
	for(i = 0;i < nout; i++) {
		if(spr_out[i].rkPref) m_pdot += (ret_idx[i] * spr_out[i].rksz);
		else m_sdot += (ret_idx[i] * spr_out[i].rksz);
	}
	if(jdim_equal) {//양측 조인 랭크가 같으면 한번에 초기화
		for(i = 0;i < njo_pre; i++) pj_idx[i] = sj_idx[i] = 0;
	} else {
		for(i = 0;i < njo_pre; i++) pj_idx[i] = 0;
		for(i = 0;i < njo_suf; i++) sj_idx[i] = 0;
	}
	/*intt j, k, M = spr_out[0].rkdim, lda = spr_pre_jo[0].rkdim, N = spr_suf_out[0].rkdim, ldb = spr_suf_jo[0].rkdim, K = lda;
	for(i = po_idx[0];i < M; i++) {
		for(j = so_idx[0];j < N; j++) {
			for(k = 0, sum = 0;k < K; k++) {
				sum += *(m_pdot + i * lda + k) * *(m_sdot + k * ldb + j);
			}
			if(rplus == 0) *(m_rdot + roff) = 0;
			else *(m_rdot + roff) *= rplus;
			*(m_rdot + roff) += sum;
			roff++;
			if(roff >= n) goto LB2;
		}
	}
	goto LB2;*/
	nout--;
	const intt njo_pre2 = njo_pre -1, njo_suf2 = njo_suf -1;
	const intt pjdim = spr_pre_jo[njo_pre2].rkdim, pjsz = spr_pre_jo[njo_pre2].rksz;
	const intt sjdim = spr_suf_jo[njo_suf2].rkdim, sjsz = spr_suf_jo[njo_suf2].rksz;
	const intt podim = spr_out[nout].rkdim, posz = spr_out[nout].rksz;
	bool lastout_is_pref = spr_out[nout].rkPref;
	intt i_po = ret_idx[nout], i_pj = 0, i_sj = 0;
	
	for(;roff < n; roff++) {
		for(sum = 0;;) {//끝에서부터 계수, 조인 인덱스만 순차적으로 증가하며 양측 원소 곱셈, 조인트 차원 인덱스 설정
			//printf("%f %f\n", *m_pdot, *m_sdot);
			sum += *m_pdot * *m_sdot;
			if(jdim_equal) {//양측 조인 인덱스가 같으면 대표로 pre join인덱스로 제어
				for(;;) {//pre 조인차원 인덱스 증가
					if(pjdim == ++i_pj) {//단위(차원) 올림과 동시에 마지막에서 조인 계수 초기화가 자동으로 된다.
						i_pj = 0;
						i = njo_pre2;
J0:;					m_pdot -= spr_pre_jo[i].rktsz;
						m_sdot -= spr_suf_jo[i].rktsz;
						if(--i < 0) goto LB1;
						else {//중간 차원 증가
							if(spr_pre_jo[i].rkdim == ++pj_idx[i]) {
								pj_idx[i] = 0;
								goto J0;
							} else {
								m_pdot += spr_pre_jo[i].rksz;//다음 상위 단위(차원)값 증가
								m_sdot += spr_suf_jo[i].rksz;
								break;
							}
						}
					} else {//말단 차원 증가
						m_pdot += pjsz;//현 단위(차원)값 증가
						m_sdot += sjsz;
						break;
					}
				}
			} else {
				for(;;) {//pre 조인차원 인덱스 증가
					if(pjdim == ++i_pj) {//단위(차원) 올림과 동시에 마지막에서 조인 계수 초기화가 자동으로 된다.
						i_pj = 0;
						i = njo_pre2;
J1:;					m_pdot -= spr_pre_jo[i].rktsz;
						if(--i < 0) break;
						else {
							if(spr_pre_jo[i].rkdim == ++pj_idx[i]) {
								pj_idx[i] = 0;
								goto J1;
							} else {
								m_pdot += spr_pre_jo[i].rksz;//다음 상위 단위(차원)값 증가
								break;
							}
						}
					} else {
						m_pdot += pjsz;//현 단위(차원)값 증가
						break;
					}
				}
				for(;;) {//suf 조인차원 인덱스 증가
					if(sjdim == ++i_sj) {
						i_sj = 0;
						i = njo_suf2;
J2:;					m_sdot -= spr_suf_jo[i].rktsz;
						if(--i < 0) goto LB1;
						else {
							if(spr_suf_jo[i].rkdim == ++sj_idx[i]) {
								sj_idx[i] = 0;
								goto J2;
							} else {
								m_sdot += spr_suf_jo[i].rksz;//다음 상위 단위(차원)값 증가
								break;
							}
						}
					} else {
						m_sdot += sjsz;
						break;
					}
				}
			}
		}
LB1:;
		if(rplus) *(m_rdot + roff) += sum;
		else *(m_rdot + roff) = sum;
		for(;;) {//출력 차원 인덱스 증가(pre와 suf의 출력 축 통합하여)
			if(podim == ++i_po) {//podim은 pre와 suf중 통합 차원에서 맨 하위에 위치하는 것의 디멘젼, i_po, posz, lastout_is_pref도 마찬가지)
				i_po = 0;
				i = nout;
O2:				if(spr_out[i].rkPref) m_pdot -= spr_out[i].rktsz;
				else m_sdot -= spr_out[i].rktsz;
				if(--i < 0) goto LB2;
				else {
					if(spr_out[i].rkdim == ++ret_idx[i]) {
						ret_idx[i] = 0;
						goto O2;
					} else {
						if(spr_out[i].rkPref) m_pdot += spr_out[i].rksz;//다음 상위 단위(차원)값 증가
						else m_sdot += spr_out[i].rksz;
						break;//suf out 중간 차원 단위 증가
					}
				}
			} else {
				if(lastout_is_pref) m_pdot += posz;
				else m_sdot += posz;
				break;
			}
		}
	}
LB2:;
	return n - idx_origin * idx_width;
}
typedef struct {
	intt trssz, trstsz, trsdim;
} TransRank;
typedef struct {
	intt ntrDims;
	intt trRankRet[MX_DIM], trTxid[MX_DIM];
	TransRank tspmap[MX_DIM];
} TransVar;
template<typename T>
intt ctrs_t(void *pcxt, T *m_strs, T *m_rtrs, intt r_size, intt idx_origin, intt idx_width, bool bw)
{
	TransVar *tsvar = (TransVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt ndims = tsvar->ntrDims, *rrank = tsvar->trRankRet, i, ridx[MX_DIM];
	TransRank *tmap = tsvar->tspmap;
	
	_offset2idx(ndims, rrank, roff, ridx);
	for(i = 0;i < ndims; i++) {
		m_strs += (ridx[i] * tmap[i].trssz);
	}
	intt it = ridx[--ndims];
	const intt tdim = tmap[ndims].trsdim, tsz = tmap[ndims].trssz;
	
	for(;roff < n; roff++) {
		if(bw) *(m_rtrs + roff) += *m_strs;
		else *(m_rtrs + roff) = *m_strs;
		for(;;) {
			if(tdim == ++it) {//tdim은 맨 하위에 위치하는 것의 디멘젼, it, tsz, lastout_is_pref도 마찬가지)
				it = 0;
				i = ndims;
LB1:			m_strs -= tmap[i].trstsz;
				if(--i < 0) goto LB2;
				else {
					if(tmap[i].trsdim == ++ridx[i]) {
						ridx[i] = 0;
						goto LB1;
					} else {
						 m_strs += tmap[i].trssz;//다음 상위 단위(차원)값 증가
						break;//중간 차원 단위 증가
					}
				}
			} else {
				m_strs += tsz;
				break;
			}
		}
	}
LB2:;
	return n - idx_origin * idx_width;
}
typedef struct {
	intt noprDims1;
	intt oprRank1[MX_DIM];
} OprVar1;
template<typename T>
intt csoftx_t(void *pcxt, T *m_ssfx, T *m_rsfx, T *m_sum, T *m_max, intt r_size, intt f_size, intt idx_origin, intt idx_width)
{
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	floatt v;

	for(;roff < n; roff++) {
		if(*(m_max + roff / f_size) < *(m_ssfx + roff)) *(m_max + roff / f_size) = *(m_ssfx + roff);
	}
	for(roff = idx_origin * idx_width;roff < n; roff++) {
		v = std::exp(*(m_ssfx + roff) - *(m_max + roff / f_size));
		*(m_sum + roff / f_size) += v;
		*(m_rsfx + roff) = v;
	}
	for(roff = idx_origin * idx_width;roff < n; roff++) {
		*(m_rsfx + roff) = *(m_rsfx + roff) / (*(m_sum + roff / f_size) + 1e-8);
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt csoftx_cross_e_t(void *pcxt, T *m_ssfx, T *m_rsfx, T *m_tsfx, intt r_size, intt f_size, intt idx_origin, intt idx_width)
{
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	for(;roff < n; roff++) {
		*(m_rsfx + (roff / f_size)) += -1.0f * std::log(*(m_ssfx + roff) + 1e-8) * *(m_tsfx + roff);
		//printf("%f %f %f\n", *(m_rsfx + (roff / f_size)), *(m_ssfx + roff), *(m_tsfx + roff));
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt cmse_f(void *pcxt, T *m_smet, T *m_tmet, T *m_rmet, intt r_size, intt idx_origin, intt idx_width, bool mean)
{
	if(mean) {//매트릭스 전체 차이 젭곱 평균
		intt roff = 0;
		for(;roff < r_size; roff++) {
			m_rmet[0] += ((m_smet[roff] - m_tmet[roff]) * (m_smet[roff] - m_tmet[roff]));
		}
		m_rmet[0] /= r_size;
		return 1;
	} else {//배치단위 차이 젭곱까지만 계산
		intt roff = idx_origin * idx_width;
		intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
		for(;roff < n; roff++) {
			m_rmet[roff] = (m_smet[roff] - m_tmet[roff]) * (m_smet[roff] - m_tmet[roff]);
		}
		return n;
	}
}
template<typename T>
intt csum_f(void *pcxt, T *m_smet, T *m_rmet, intt r_size, T *cmul, bool mean) //전체 합, 평균 계산
{
	intt roff = 0;

	for(;roff < r_size; roff++) {
		*m_rmet += *(m_smet + roff);
	}
	if(cmul) *m_rmet *= *(T *)cmul;
	if(mean) *m_rmet /= r_size;
	return r_size;
}
typedef struct {
	intt nrkPre, nrkSuf, nrkOut, ovpad;//ovpad는주소 8바이트 정렬하기위해 
	intt dimPre[MX_DIM], dimSuf[MX_DIM], dimOut[MX_DIM];
	intt rankPre[MX_DIM], rankSuf[MX_DIM], rankOut[MX_DIM];
	intt idxOne[MX_DIM * 4];
} OneVar;
template<typename T>
intt cbatch_sum_t(void *pcxt, T *mpre, T *mret, intt r_size, intt idx_origin, intt idx_width, bool bw, sytet rplus)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt sum_sz = ovar->idxOne[1];

	if(bw) {
		floatt div = *(floatt *)&ovar->idxOne[2];
		if(ovar->idxOne[0]) {
			if(rplus) {
				for(;roff < n; roff++) {//roff는 mret옵셋
					mret[roff] += mpre[roff / sum_sz] * div;
				}
			} else {
				for(;roff < n; roff++) {
					mret[roff] = mpre[roff / sum_sz] * div;
				}
			}
		} else {
			if(rplus) {
				for(;roff < n; roff++) {
					mret[roff] += mpre[roff / sum_sz];
				}
			} else {
				for(;roff < n; roff++) {
					mret[roff] = mpre[roff / sum_sz];
				}
			}
		}
	} else {
		intt beg;
		for(beg = roff;roff < n; roff++) mret[roff / sum_sz] += mpre[roff];//roff는 mpre옵셋
		if(ovar->idxOne[0]) {//평균 구하기, 이번 mpre의 시작과 끝 구간을 sum_sz(배치차원 이외 사이즈 합)단위로 
			intt n2 = n / sum_sz;//줄어든 배치 평균이 적재된 mret의 시작과 끝 옵셋으로 변환하여 이 구간을 평균값 산출
			for(roff = beg / sum_sz;roff < n2; roff++) mret[roff] /= sum_sz;
		}
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt coptadm_t(void *pcxt, T *mm, T *mv, T *mg, T *mr, intt r_size, intt idx_origin, intt idx_width,
	floatt beta1, floatt beta2, floatt lr, floatt ep, intt dec)
{
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	for(;roff < n; roff++) {
		mm[roff] += (1.0f - beta1) * (mg[roff] - mm[roff]);//우항: mm[roff] + (l - beta1)*mg[roff] - mm[roff] + mm[roff]*beta1
														//	  = mm[roff]*beta1 + (l - beta1)*mg[roff]
		mv[roff] += (1.0f - beta2) * (mg[roff] * mg[roff] - mv[roff]);
		mr[roff] += dec * lr * mm[roff] / (std::sqrt(mv[roff]) + ep);
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt coptsgd_t(void *pcxt, T *mg, T *mr, intt r_size, intt idx_origin, intt idx_width, floatt lr, intt dec)
{
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	for(;roff < n; roff++) {
		mr[roff] += dec * lr * mg[roff];
	}
	return n - idx_origin * idx_width;
}

#define DACTF_TANH	1
#define DACTF_RELU	3
#define DACTF_SIGM	5
#define DACTF_LRELU	7

#define MATH_SQRT	8
#define DMATH_SQRT	9

#define DJUST_COPY	10
#define DJUST_COPY2	11

#define MATH_LOG	12
#define DMATH_LOG	13
#define DLOSS_FUNC	14
#define SCOOP_UP	15
#define DSCOOP_UP	16
#define MINMAX_NORMAL	17
#define DMINMAX_NORMAL	18

template<typename T>
intt cactf_t(void *pcxt, T *mpre, T *msuf, T *mret, intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet rplus)
{//mpre는 순전파때 출력, msuf는 상위 기울기
	OneVar *ovar = (OneVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	T v;

	switch(aop2) {
	case ACTF_TANH:
		for(; roff < n; roff++) {
			mret[roff] = std::tanh(mpre[roff]);
			//printf("%f %f\n", mret[roff], mpre[roff]);
		}
		break;
	case DACTF_TANH:
		if(rplus) {
			for(; roff < n; roff++) {
				mret[roff] += ((1.0 - std::tanh(mpre[roff]) * std::tanh(mpre[roff])) * msuf[roff]);
			}
		}
		else {
			for(; roff < n; roff++) {
				mret[roff] = (1.0 - std::tanh(mpre[roff]) * std::tanh(mpre[roff])) * msuf[roff];
			}
		}
		break;
	case ACTF_RELU:
		for(; roff < n; roff++) {
			mret[roff] = mpre[roff] > 0.0 ? mpre[roff] : 0.0;
		}
		break;
	case DACTF_RELU:
		if(rplus) {
			for(; roff < n; roff++) {
				mret[roff] += ((mpre[roff] > 0.0 ? 1.0 : 0.0) * msuf[roff]);
			}
		}
		else {
			for(; roff < n; roff++) {
				mret[roff] = (mpre[roff] > 0.0 ? 1.0 : 0.0) * msuf[roff];
			}
		}
		break;
	case ACTF_SIGM:
		for(; roff < n; roff++) {
			mret[roff] = 1.0 / (1.0 + std::exp(-mpre[roff]));//1.0f/(1.0f + std::exp(-a));
		}
		break;
	case DACTF_SIGM:
		if(rplus) {
			for(; roff < n; roff++) {
				v = 1.0 / (1.0 + std::exp(-mpre[roff]));
				mret[roff] += (1.0 - v) * v * msuf[roff];
			}
		}
		else {
			for(; roff < n; roff++) {
				v = 1.0 / (1.0 + std::exp(-mpre[roff]));
				mret[roff] = (1.0 - v) * v * msuf[roff];
			}
		}
		break;
	case ACTF_LRELU:
	{
		T m = *(T *)&ovar->idxOne[0];
		for(; roff < n; roff++) {
			mret[roff] = mpre[roff] > 0.0 ? mpre[roff] : m * mpre[roff];
		}
	}
		break;
	case DACTF_LRELU:
	{
		T m = *(T *)&ovar->idxOne[0];
		if(rplus) {
			for(; roff < n; roff++) {
				mret[roff] += ((mpre[roff] > 0.0 ? 1.0 : m) * msuf[roff]);
			}
		} else {
			for(; roff < n; roff++) {
				mret[roff] = (mpre[roff] > 0.0 ? 1.0 : m) * msuf[roff];
			}
		}
	}
		break;
	case MATH_SQRT:
		for(; roff < n; roff++) {
			mret[roff] = std::sqrt(mpre[roff]);
		}
		break;
	case DMATH_SQRT:
		if(rplus) {
			for(; roff < n; roff++) {
				mret[roff] += ((0.5 * 1.0 / sqrt(mpre[roff] + 1e-9)) * msuf[roff]);//0.5 * pow(mpre[roff], -0.5f)
			}
		}
		else {
			for(; roff < n; roff++) {
				mret[roff] = (0.5 * 1.0 / sqrt(mpre[roff] + 1e-9)) * msuf[roff];//0.5 * pow(mpre[roff], -0.5f)
			}
		}
		break;
	case DJUST_COPY://mpre는 상위 기울기, 단순하 역전파때 기울기를 더하기위해 사용.
		if(msuf) {
			if(rplus) {
				for(; roff < n; roff++) {
					mret[roff] += mpre[roff] * msuf[roff];
				}
			}
			else {
				for(; roff < n; roff++) {
					mret[roff] = mpre[roff] * msuf[roff];
				}
			}
		}
		else {
			if(rplus) {
				for(; roff < n; roff++) {
					mret[roff] += mpre[roff];
				}
			}
			else {
				for(; roff < n; roff++) {
					mret[roff] = mpre[roff];
				}
			}
		}
		break;
	case DJUST_COPY2://mpre는 상위 기울기, mpre, msuf가 원소 1개
		if(msuf) {
			if(rplus) {
				for(; roff < n; roff++) {
					mret[roff] += mpre[0] * msuf[0];
				}
			}
			else {
				for(; roff < n; roff++) {
					mret[roff] = mpre[0] * msuf[0];
				}
			}
		}
		else {
			if(rplus) {
				for(; roff < n; roff++) {
					mret[roff] += mpre[0];
				}
			}
			else {
				for(; roff < n; roff++) {
					mret[roff] = mpre[0];
				}
			}
		}
		break;
	case MATH_LOG:
		for(; roff < n; roff++) {
			mret[roff] = std::log(mpre[roff]);
		}
		break;
	case DMATH_LOG:
		for(; roff < n; roff++) {
			mret[roff] = (1.0 / (mpre[roff] + 1e-9)) * msuf[roff];
			//printf("%f %f %f\n", mret[roff], mpre[roff], msuf[roff]);
		}
		break;
	case DLOSS_FUNC:
		for(; roff < n; roff++) {//msuf는 목표값
			mret[roff] += ((mpre[roff] - msuf[roff]) / *(T *)ovar->idxOne);//배치사이즈로 나눔
		}
		break;
	case SCOOP_UP:
	{
		intt slidex, slidey, stridex, stridey, outx, outy, ibatch, rest;
		intt prey = ovar->idxOne[6], prex = ovar->idxOne[7], sz_feat = ovar->idxOne[8], d2;
		slidey = ovar->idxOne[0]; slidex = ovar->idxOne[1] * sz_feat; stridey = ovar->idxOne[2];
		stridex = ovar->idxOne[3]; outy = ovar->idxOne[4]; outx = ovar->idxOne[5];
		intt n_derive = outx * outy, sz_slide = slidex * slidey, irow, icol, i, j;
		intt sz_derive = n_derive * sz_slide, sz_derive_row = outx * sz_slide;
		T *px, *py, *pbatch, *pslide;
		//roff가 스트라이드에 의해 확장 파생된 매트릭스의 옵셋이므로 이로서 인덱스를 획득하고 이로부터 소스의 옵셋을 계산한다.
		ibatch = roff / sz_derive;//스트라이드에 의해 파생된 매트릭스에서의 이번 옵셋의 시작 배치로우 인덱스
		rest = roff - (ibatch * sz_derive);//파생된 매트릭스의 이번 시작 배치로우의 남은 사이즈
		//스트라이드에 의해 확장된 파생 매트릭스내의 현 옵셋을 파생 매트릭스상에서의 배치인덱스와 슬라이드의 왼쪽 모서리를
		//오리진으로 하여 배치내에서의 로우와 컬럼 인덱스로 변환한다.
		if(outy != 1) {//2d
			irow = rest / sz_derive_row;//파생매트릭에서 남은 하나의 배치내의 옵셋을 파생매트릭의 한 행의 사이즈로 나눠
			rest -= irow * sz_derive_row;//파생매트릭상의 현 배치의 로우 인덱스를 구하고 이 로우의 시작 옵셋으로부터 남은
			d2 = 1;						//사이즈를 구한다.
		}
		else {
			irow = 0;//1d	
			d2 = 0;
		}
		icol = rest / sz_slide;//파생매트릭상의 남은 사이즈로부터 슬라이드 사이즈단위 컬럼 인덱스를 계산
		rest -= icol * sz_slide;

		prex *= sz_feat;
		stridey *= prex;//로우 스트라이드를 mpre매트릭스상의 값으로 변환
		stridex *= sz_feat;//컬럼 스트라이드를 mpre매트릭스상의 값으로 변환
		outy *= stridey;//한계값을 mpre매트릭스상의 값으로 변환, 제로패딩 된다면 prey에 스트라이드가 한번 더 더해진 길이 일수있다. 
		outx *= stridex;//한계값을 mpre매트릭스상의 값으로 변환, 제로패딩 된다면 prex에 스트라이드가 한번 더 더해진 길이 일수있다.
		irow *= stridey;//한계값을 mpre매트릭스상의 인젝스로 변환했으므로 초기값도 변환
		icol *= stridex;//한계값을 mpre매트릭스상의 인젝스로 변환했으므로 초기값도 변환


		if(slidey != 1) {//2d
			i = rest / slidex;//파생매트릭상의 슬라이드내의 로우 인덱스 계산 
			j = rest % slidex;//파생매트릭상의 슬라이드내의 컬럼 인덱스 계산
		}
		else {//1d
			i = 0;
			j = rest;
		}
		slidey *= prex;//한계값을 mpre매트릭스상의 값으로 변환
		i *= prex;//한계값을 mpre매트릭스상의 인젝스로 변환했으므로 초기값도 변환(슬라이드내의 로우는
		j *= sz_feat;	//mpre에서는 행 사이즈 단위로 증가)
		prey *= prex;//1개 배치사이즈
		for(;; ibatch++) {
			pbatch = mpre + ibatch * prey;//mpre매트릭스상의 이번 배치 시작 포인터 계산.
			for(; irow < outy; irow += stridey) {//파생로우인덱스는 소스에서 로우 스트라이드 단위로 증가됨
				py = pbatch + d2 * irow;//d2는 1d이면 0, mpre매트릭스상의 로우포인터 설정.
				for(; icol < outx; icol += stridex) {//파생컬럼인덱스는 소스에서 컬럼 스트라이드 단위로 증가됨
					pslide = py + icol;//슬라이드 오리진(좌측 상위) 포인터 설정
					for(; i < slidey; i += prex) {//슬라이드내의 로우 하나씩 증가, 슬라이드내의 로우는
						px = pslide + i;//mpre에서는 행 사이즈 단위로 증가해야한다.
						for(; j < slidex; j++) {//슬라이드내의 컬럼 하나씩 증가
							if(roff < n) mret[roff++] = (irow + i >= prey || icol + j >= prex ? 0 : *(px + j));//제로패딩
							else return n - idx_origin * idx_width;
							//printf("%f\n", mret[roff - 1]);
						}
						j = 0;
					}
					i = 0;
				}
				icol = 0;//파생매트릭의 다음 로우부터는 컬럼을 0부터 시작
			}
			irow = 0;//파생매트릭의 다음 배치부터는 로우를 0부터 시작
		}
	}
	break;
	case DSCOOP_UP:
		break;
	case MINMAX_NORMAL:
	{
		T min = (T)*(doublet *)&ovar->idxOne[0], diffv = (T)*(doublet *)&ovar->idxOne[2];
		for(; roff < n; roff++) {
			mret[roff] = (mpre[roff] - min) / diffv;
		}
	}
	break;
	case DMINMAX_NORMAL:
		break;
	}
	return n - idx_origin * idx_width;
}
#define ACTF_PRELU	0
#define DACTF_PRELU	1
template<typename T>
intt cactf2_t(void *pcxt, T *mpre, T *msuf, T *mret, T *m1, T *m2, intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet rplus)
{
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	switch(aop2) {
	case ACTF_PRELU://msuf는 prelu 가중치 데이터
		for(;roff < n; roff++) {
			mret[roff] = mpre[roff] > 0.0 ? mpre[roff] : msuf[roff] * mpre[roff];// x > 0.0f ? x : a * x
		}
		break;
	case DACTF_PRELU://mpre는 상위 기울기, msuf는 순전파때 출력, mret는 순전파때 입력 기울기, 
	{				//m1 - prelu 가중치 데이터, m2 - prelu 가중치 기울기
		OneVar *ovar = (OneVar *)pcxt;
		if(rplus) {
			for(;roff < n; roff++) {
				mret[roff] += ((msuf[roff] > 0.0 ? 1.0 : m1[roff]) * mpre[roff]);//dst[idx] = src[idx] > 0.0f ? 1.0f : a[idx];
				m2[roff] += ((msuf[roff] > 0.0 ? 0.0 : msuf[roff]) * mpre[roff]);//da[idx] = src[idx] > 0.0f ? 0.0f : src[idx];
			}
		} else {
			for(;roff < n; roff++) {
				mret[roff] = (msuf[roff] > 0.0 ? 1.0 : m1[roff]) * mpre[roff];//dst[idx] = src[idx] > 0.0f ? 1.0f : a[idx];
				m2[roff] = (msuf[roff] > 0.0 ? 0.0 : msuf[roff]) * mpre[roff];//da[idx] = src[idx] > 0.0f ? 0.0f : src[idx];
			}
		}
		/*
		if(rplus) {
			for(;roff < n; roff++) {
				mret[roff] += (mpre[roff] > 0.0 ? 1.0 : msuf[roff]);//dst[idx] = src[idx] > 0.0f ? 1.0f : a[idx];
				rsuf[roff] += (mpre[roff] > 0.0 ? 0.0 : mpre[roff]);//da[idx] = src[idx] > 0.0f ? 0.0f : src[idx];
			}
		} else {
			for(;roff < n; roff++) {
				mret[roff] = mpre[roff] > 0.0 ? 1.0 : msuf[roff];//dst[idx] = src[idx] > 0.0f ? 1.0f : a[idx];
				rsuf[roff] = mpre[roff] > 0.0 ? 0.0 : mpre[roff];//da[idx] = src[idx] > 0.0f ? 0.0f : src[idx];
			}
		}*/
	}
		break;
	}
	return n - idx_origin * idx_width;
}
#define TWOF_SQDIFF	0
#define DTWOF_SQDIFF 1
template<typename T>
intt ctwo_t(void *pcxt, T *mpre, T *msuf, T *mret, T *bpre, T *bsuf, intt r_size, intt idx_origin, intt idx_width, intt aop2)
{
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	switch(aop2) {
	case TWOF_SQDIFF:
		for(;roff < n; roff++) {
			mret[roff] = (mpre[roff] - msuf[roff]) * (mpre[roff] - msuf[roff]);
		}
		break;
	case DTWOF_SQDIFF:
		for(;roff < n; roff++) {
			mpre[roff] += 2 * mret[roff];
			msuf[roff] += -2 * mret[roff];
		}
		break;
	}
	return n - idx_origin * idx_width;
}
#define int_val_type(ival, sval, tp) {\
	switch(tp) {\
	case tshort:\
		ival = *(shortt *)sval;\
		break;\
	case tfloat:\
		ival = *(floatt *)sval;\
		break;\
	case tint:\
		ival = *(intt *)sval;\
		break;\
	case tlong:\
		ival = *(longt *)sval;\
		break;\
	case tdouble:\
		ival = *(doublet *)sval;\
		break;\
	}\
}
template<typename T>
intt cembedding_t(T *mpre, T *msuf, T *mret, intt r_size, intt idx_origin, intt idx_width, intt sz_embed, 
	intt stp, intt etable_sz, intt bw)
{
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size), idx;

	if(bw) {//msuf - input, mret - lookup table, mpre - embeded, roff는 mpre 기준
		switch(stp) {
		case tshort:
			for(;roff < n; roff++) {
				idx = *(shortt *)&msuf[roff / sz_embed];
				if(idx >= etable_sz) throwFault(-1, "embedding table size over\n");
				mret[idx*sz_embed + roff % sz_embed] += mpre[roff];
			}
			break;
		case tfloat:
			for(;roff < n; roff++) {
				idx = *(floatt *)&msuf[roff / sz_embed];
				if(idx >= etable_sz) throwFault(-1, "embedding table size over\n");
				mret[idx*sz_embed + roff % sz_embed] += mpre[roff];
				//printf("%d %d %f %f\n", roff, idx*sz_embed + roff % sz_embed, mpre[roff], mret[idx*sz_embed + roff % sz_embed], mpre[roff]);
			}
			break;
		case tint:
			for(;roff < n; roff++) {
				idx = *(intt *)&msuf[roff / sz_embed];
				if(idx >= etable_sz) throwFault(-1, "embedding table size over\n");
				mret[idx*sz_embed + roff % sz_embed] += mpre[roff];			}
			break;
		case tlong:
			for(;roff < n; roff++) {
				idx = *(longt *)&msuf[roff / sz_embed];
				if(idx >= etable_sz) throwFault(-1, "embedding table size over\n");
				mret[idx*sz_embed + roff % sz_embed] += mpre[roff];			}
			break;
		case tdouble:
			for(;roff < n; roff++) {
				idx = *(doublet *)&msuf[roff / sz_embed];
				if(idx >= etable_sz) throwFault(-1, "embedding table size over\n");
				mret[idx*sz_embed + roff % sz_embed] += mpre[roff];			}
			break;
		}
	} else {//msuf - input, mret - embeded, mpre - lookup table, roff는 mret 기준
		switch(stp) {
		case tshort:
			for(;roff < n; roff++) {
				idx = *(shortt *)&msuf[roff / sz_embed];
				if(idx >= etable_sz) throwFault(-1, "embedding table size over\n");
				mret[roff] = mpre[idx*sz_embed + roff % sz_embed];
			}
			break;
		case tfloat:
			for(;roff < n; roff++) {
				idx = *(floatt *)&msuf[roff / sz_embed];
				if(idx >= etable_sz) throwFault(-1, "embedding table size over\n");
				mret[roff] = mpre[idx*sz_embed + roff % sz_embed];
				//printf("%d %d %f %f\n", roff, idx*sz_embed + roff % sz_embed, mpre[roff], mret[idx*sz_embed + roff % sz_embed], mpre[roff]);
			}
			break;
		case tint:
			for(;roff < n; roff++) {
				idx = *(intt *)&msuf[roff / sz_embed];
				if(idx >= etable_sz) throwFault(-1, "embedding table size over\n");
				mret[roff] = mpre[idx*sz_embed + roff % sz_embed];
			}
			break;
		case tlong:
			for(;roff < n; roff++) {
				idx = *(longt *)&msuf[roff / sz_embed];
				if(idx >= etable_sz) throwFault(-1, "embedding table size over\n");
				mret[roff] = mpre[idx*sz_embed + roff % sz_embed];
			}
			break;
		case tdouble:
			for(;roff < n; roff++) {
				idx = *(doublet *)&msuf[roff / sz_embed];
				if(idx >= etable_sz) throwFault(-1, "embedding table size over\n");
				mret[roff] = mpre[idx*sz_embed + roff % sz_embed];
			}
			break;
		}
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt conehot_t(void *pcxt, T *mpre, T *mret, intt r_size, intt idx_origin, intt idx_width)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt poff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	T onv = (T)*(doublet *)ovar->idxOne;
	intt roff, psz = ovar->idxOne[4], depth = ovar->idxOne[5];

	for(;poff < n; poff++) {
		if(*(mpre + poff) >= depth || *(mpre + poff) < 0) continue;
		roff = (poff / psz) * depth * psz + *(mpre + poff) * psz + poff % psz;
		*(mret + roff) = onv;
	}
	return n - idx_origin * idx_width;
}
typedef struct {
	intt slsz, sltsz, sldim, slbase;
} SliceRank;
template<typename T>
intt cslice_t(void *pcxt, T *mpre, T *mret, intt r_size, intt idx_origin, intt idx_width, bool bw, sytet rplus)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	SliceRank *slicer = (SliceRank *)ovar->idxOne;
	intt *srank, ndims = ovar->nrkPre;
	intt ridx[MX_DIM], i;
	T *cmem, *smem;

	srank = ovar->rankOut;
	if(bw) {
		cmem = mret;
		smem = mpre;//slice matrix
	} else {
		cmem = mpre;
		smem = mret;//slice matrix
	}
	_offset2idx(ndims, srank, roff, ridx);
	for(i = 0;i < ndims; i++) {
		cmem += (slicer[i].slbase + ridx[i] * slicer[i].slsz);
	}
	intt it = ridx[--ndims];
	const intt tdim = slicer[ndims].sldim, tsz = slicer[ndims].slsz;

	for(;roff < n; roff++) {
		if(bw) {
			if(rplus) *cmem += *(smem + roff);
			else *cmem = *(smem + roff);
		} else *(smem + roff) = *cmem;
		for(;;) {
			if(tdim == ++it) {//tdim은 맨 하위에 위치하는 것의 디멘젼, it, tsz도 마찬가지)
				it = 0;
				i = ndims;
LB1:			cmem -= slicer[i].sltsz;
				if(--i < 0) goto LB2;
				else {
					if(slicer[i].sldim == ++ridx[i]) {
						ridx[i] = 0;
						goto LB1;
					} else {
						cmem += slicer[i].slsz;//다음 상위 단위(차원)값 증가
						break;//중간 차원 단위 증가
					}
				}
			} else {
				cmem += tsz;
				break;
			}
		}
	}
LB2:;
	return n - idx_origin * idx_width;
}
template<typename T>
intt cargmax_t(void *pcxt, T *mpre, T *mret, intt r_size, intt idx_origin, intt idx_width)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt poff, i, naxis = ovar->nrkPre, inner_sz = ovar->nrkSuf, outer_sz = ovar->nrkOut;
	T vmax;

	for(;roff < n; roff++) {
		poff = (roff / inner_sz) * outer_sz + roff % inner_sz;//roff는 poff에서 max축이 없어지고 축소된 옵셋이므로
		//roff를 inner_sz로 나눈 인덱스는 poff기준으로 outer_sz로 나눈 단위와 같다. 따라서 이 인덱스에 outer_sz를 곱해서
		//축소전 outer_sz단위 poff를 구하고 여기에 inner_sz내의 옵셋을 더해 시작 옵셋 poff를 획득한다.
		vmax = *(mpre + poff);
		*(mret + roff) = 0;
		for(i = 0;i < naxis; i++, poff += inner_sz) {
			if(vmax < *(mpre + poff)) {
				vmax = *(mpre + poff);
				*(mret + roff) = i;
			}
		}
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt cvmax_t(void *pcxt, T *mpre, T *msuf, T *mret, intt r_size, intt idx_origin, intt idx_width, intt bw, sytet rplus)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt poff, i, naxis = ovar->nrkPre, inner_sz = ovar->nrkSuf, outer_sz = ovar->nrkOut;
	//vmax단계에서는 분기될 일 없으므로 rplus는 무시
	if(bw) {//mpre는 arg max index map, msuf는 역전파 기울기, 둘다 poff(축소 옵셋)으로 동일, mret는 소스, roff는 풀링 소스 옵셋
		for(; roff < n; roff++) {
			poff = (roff / outer_sz) * inner_sz + roff % inner_sz;//옵셋 축소
			if(*(mpre + poff) == (roff % outer_sz) / inner_sz) *(mret + roff) = *(msuf + poff);
			else *(mret + roff) = 0;
			//printf("%d %d %f %f %f\n", poff, roff, *(mpre + poff), *(msuf + poff), *(mret + roff));
		}
	} else {//mpre는 소스, msuf는 arg max index map, mret는 max 결과, roff는 풀링 결과 옵셋, poff는 풀링 소스 확장 옵셋
		for(; roff < n; roff++) {
			poff = (roff / inner_sz) * outer_sz + roff % inner_sz;//roff는 poff에서 max축이 없어지고 축소된 옵셋이므로
			//roff를 inner_sz로 나눈 인덱스는 poff기준으로 outer_sz로 나눈 단위와 같다. 따라서 이 인덱스에 outer_sz를 곱해서
			//축소전 outer_sz단위 poff를 구하고 여기에 inner_sz내의 옵셋을 더해 시작 옵셋 poff를 획득한다.
			*(mret + roff) = *(mpre + poff);
			*(msuf + roff) = 0;
			for(i = 0; i < naxis; i++, poff += inner_sz) {
				if(*(mret + roff) < *(mpre + poff)) {
					*(mret + roff) = *(mpre + poff);
					*(msuf + roff) = i;
				}
			}
		}
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt cequal_t(void *pcxt, T mpre[], T msuf[], T mret[], intt r_size, intt idx_origin, intt idx_width)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	bool eq = ovar->idxOne[0], cscalr = ovar->idxOne[1];
	T csv = *(T *)&ovar->idxOne[2];

	for(;roff < n; roff++) {
		if(cscalr) {
			if(eq) {
				if(mpre[roff] == csv) mret[roff] = (T)1;
				else mret[roff] = (T)0;
			} else {
				if(mpre[roff] == csv) mret[roff] = (T)0;
				else mret[roff] = (T)1;
			}
		} else {
			if(eq) {
				if(mpre[roff] == msuf[roff]) mret[roff] = (T)1;
				else mret[roff] = (T)0;
			} else {
				if(mpre[roff] == msuf[roff]) mret[roff] = (T)0;
				else mret[roff] = (T)1;
			}
		}
	}
	return n - idx_origin * idx_width;
}
#define TYPE1_CLIP	0
#define DIAGO_MUL	1
#define DIAGO_FILL	2
template<typename T>
intt ctype1_t(void *pcxt, T mpre[], T msuf[], T mret[], intt r_size, intt idx_origin, intt idx_width, intt aop2)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	switch(aop2) {
	case TYPE1_CLIP:
	{
		doublet low = *(doublet *)&ovar->idxOne[0], high = *(doublet *)&ovar->idxOne[2];
		for(;roff < n; roff++) {
			if(mpre[roff] < low) mret[roff] = low;
			else if(mpre[roff] > high) mret[roff] = high;
			else mret[roff] = mpre[roff];
		}
	}
		break;
	case DIAGO_MUL:
	{
		intt dimen = ovar->idxOne[0];
		intt d = roff / dimen, r = roff % dimen;
		intt poff = d * dimen * dimen + r * dimen + r;
		for(intt i = 0;roff < n; roff++) {
			mret[roff] = mpre[poff] * msuf[poff];
			if(++r == dimen) {
				r = 0;
				poff++;
			} else poff += dimen + 1;
		}
	}
		break;
	case DIAGO_FILL:
	{
		intt dimen = ovar->idxOne[0];
		intt d = roff / dimen, r = roff % dimen;
		intt poff = d * dimen * dimen + r * dimen + r;
		for(intt i = 0;roff < n; roff++) {
			mret[poff] = mpre[roff];
			if(++r == dimen) {
				r = 0;
				poff++;
			} else poff += dimen + 1;
		}
	}
		return (n - idx_origin * idx_width) * -1;//roff, r_size가 mret가 아닌 mpre의 것이므로 리턴하여 매트릭스 전체 복사하게 한다.
	}
	return n - idx_origin * idx_width;
}
#define RAND_T_N	0
#define RAND_T_U	1
#define RAND_T_L	2
#define RAND_T_P	3
template<typename T, typename T2>
intt crandom_t(void *pcxt, T mpre[], T msuf[], T mret[], intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet t, intt seed)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	switch(aop2) {
	case RAND_T_N:
	{
		doublet m = *(doublet *)&ovar->idxOne[0], a = *(doublet *)&ovar->idxOne[2];
		random_device rd;//운영체제에서 시간, 프로세스 아이디, 환경변수, 유저이름, 컴푸터 이름등을 해시하여 SHA-1함수로 시드생성.
		mt19937 mt(seed < 0 ? rd() : seed);
		//mt19937 mt(1729);//1729라는 시드값으로 난수 생성.
		normal_distribution<T2> initd1(m, a);
		for(;roff < n; roff++) {
			mpre[roff] = (T)initd1(mt);
		}
	}
	break;
	case RAND_T_U:
	{
		doublet m = *(doublet *)&ovar->idxOne[0], a = *(doublet *)&ovar->idxOne[2];
		random_device rd;
		mt19937 mt(seed < 0 ? rd() : seed);
		if(t == tfloat || t == tdouble) {
			uniform_real_distribution<T2> initd1(m, a);
			for(;roff < n; roff++) {
				mpre[roff] = (T)initd1(mt);
			}
		} else {
			uniform_int_distribution<intt> initd1(m, a);
			for(;roff < n; roff++) {
				mpre[roff] = (T)initd1(mt);
			}
		}
	}
	break;
	}
	return n - idx_origin * idx_width;
}
template<typename T>
intt clayer_norm_t(void *pcxt, T *mi, T *mr, T *md, T *mz, T *mv, T *mean, T *g_mz, T *var, T *mdmean,
	T *ga, T *be, T *g_gm, T *g_be, intt r_size, intt idx_origin, intt idx_width, intt dsz, bool bw)
{
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt i, q, r;//m = n / dsz;
	T v;

	if(bw) {
		/*for(i = roff;i < n; i++) g_mz[i] = mi[i] * ga[i % dsz];//상위 구배값에 감마값을 곱하여 순전파 출력 표준값의 구배 산출
		for(i = roff;i < n; i++) var[i / dsz] += -0.5 * g_mz[i] * md[i];//표준값 구배에 순전파 입력값 평균편차를 곱한 합
		for(i = roff;i < m; i++) var[i] *= (mv[i] * mv[i] * mv[i]);//위 합에 입력값 표준편차 역수를 곲하여 분산 구배 산출
		for(i = roff;i < n; i++) mdmean[i / dsz] += md[i];//입력값 평균 편차 합
		for(i = roff;i < m; i++) mdmean[i] /= dsz;//md mean, 입력값 평균 편차 합 평균
		for(i = roff;i < n; i++) mean[i / dsz] += -1.0 * g_mz[i] * mv[i / dsz];//순전파 출력 표준값의 구배에 표준편차 역수를 곲한 합
		for(i = roff;i < m; i++) mean[i] += -2.0 * var[i] * mdmean[i];//위 합에 분산 구배와 평균편차 평균을 곱한 값을 합계하여 평균 구배 산출
		for(i = roff;i < n; i++) mr[i] = g_mz[i] * mv[i / dsz] + (2.0 / dsz) * var[i / dsz] * md[i] + (1.0 / dsz) * mean[i / dsz];
		for(i = roff;i < n; i++) {
			g_gm[i % dsz] += mi[i] * mz[i];//상위 구배값에 순전파 출력 표준값을 곱한 값을 합하여 감마 구배 산출
			g_be[i % dsz] += mi[i];//상위 구배값을 합하여 배타 구배 산출
		}*/
		for(i = roff;i < n; i++) {
			g_mz[i] = mi[i] * ga[i % dsz];//상위 구배값에 감마값을 곱하여 순전파 출력 표준값의 구배 산출
			q = i / dsz;
			var[q] += -0.5 * g_mz[i] * md[i];//표준값 구배에 순전파 입력값 평균편차를 곱한 합
			if((i + 1) % dsz == 0) var[q] *= (mv[q] * mv[q] * mv[q]);//위 합에 입력값 표준편차 역수를 곲하여 분산 구배 산출
		}
		for(i = roff;i < n; i++) {
			q = i / dsz;
			mdmean[q] += md[i];//입력값 평균 편차 합
			if((i + 1) % dsz == 0) mdmean[q] /= dsz;//md mean, 입력값 평균 편차 합 평균
		}
		for(i = roff;i < n; i++) {
			q = i / dsz;
			mean[q] += -1.0 * g_mz[i] * mv[q];//순전파 출력 표준값의 구배에 표준편차 역수를 곲한 합
			if((i + 1) % dsz == 0) mean[q] += -2.0 * var[q] * mdmean[q];//위 합에 분산 구배와 평균편차 평균을 곱한 값을 합계하여 평균 구배 산출
		}
		for(i = roff;i < n; i++) {
			q = i / dsz;
			mr[i] = g_mz[i] * mv[q] + (2.0 / dsz) * var[q] * md[i] + (1.0 / dsz) * mean[q];
			r = i % dsz;
			g_gm[r] += mi[i] * mz[i];//상위 구배값에 순전파 출력 표준값을 곱한 값을 합하여 감마 구배 산출
			g_be[r] += mi[i];//상위 구배값을 합하여 배타 구배 산출
		}
	} else {
		/*for(i = roff;i < n; i++) mean[i / dsz] += mi[i];//입력값 합계
		for(i = roff;i < m; i++) mean[i] /= dsz;//입력값 평균
		for(i = roff;i < n; i++) {
			md[i] = v = (mi[i] - mean[i / dsz]);//평균 편차
			mv[i / dsz] += v * v;//분산, 평균편차 제곱 합
		}
		for(i = roff;i < m; i++) mv[i] = 1.0 / std::sqrt(mv[i] + 1e-9);//표준편차 역수
		for(i = roff;i < n; i++) mz[i] = md[i] * mv[i / dsz];//입력값 평균 편차에 표준편차 역수를 곱하여 표준값 산출
		for(i = roff;i < n; i++) mr[i] = mz[i] * ga[i % dsz] + be[i % dsz];//표준값에 감마를 곱하고 베타를 더하여 표준 출력값 산출.*/
		for(i = roff;i < n; i++) {
			q = i / dsz;
			mean[q] += mi[i];//입력값 합계
			if((i + 1) % dsz == 0) mean[q] /= dsz;//입력값 평균
		}
		for(i = roff;i < n; i++) {
			q = i / dsz;
			md[i] = v = (mi[i] - mean[q]);//평균 편차
			mv[q] += v * v;//분산, 평균편차 제곱 합
			if((i + 1) % dsz == 0) mv[q] = 1.0 / std::sqrt(mv[q] / dsz + 1e-9);//표준편차(분산평균) 역수
		}
		for(i = roff;i < n; i++) {
			mz[i] = md[i] * mv[i / dsz];//입력값 평균 편차에 표준편차 역수를 곱하여 표준값 산출
			r = i % dsz;
			mr[i] = mz[i] * ga[r] + be[r];//표준값에 감마를 곱하고 베타를 더하여 표준 출력값 산출.
		}
	}
	return n - idx_origin * idx_width;
}
#define LAST_OVER_SUF_SZ	N -1
template<typename DT>
intt matmul_t(void *pcxt, DT mpre[], DT msuf[], DT mret[], intt r_size,
	intt M, intt K, intt N, intt T, bool rplus, intt idx_origin, intt idx_width)
{
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt sz = M * N;//전치의 경우 K, N은 전치된 후의 인덱스이고 suf매트릭스는 전치되지 않은 원래의 매트릭스(N, K)
	intt pre_sz = M * K;//한개 배치의 pre 매트릭스 사이즈
	intt suf_sz = K * N;//한개 배치의 suf 매트릭스 사이즈
	intt pre_n = (roff / sz) * pre_sz;//pre 배치 옵셋, 배치인덱스(roff / sz) * pre 매트릭스 사이즈
	intt j = (roff / sz) * suf_sz;//suf 배치 옵셋, 배치인덱스(roff / sz) * suf 매트릭스 사이즈
	intt suf_n, i, k;
	DT sum;
	switch(T) {
	case 0://AB
		pre_n += pre_sz;
		i = (roff / N) * K;//ret의 로우 인덱스는 pre의 로우 인덱스, 배치 전체 범위에서 pre와 ret의 로우 인덱스 일대일 매칭
		suf_n = j + suf_sz + LAST_OVER_SUF_SZ;
		j += roff % N;//초기 suf컬럼옵셋, suf 배치 옵셋 + ret/suf컬럼 인덱스(roff % N) //ret와 suf의 컬럼인덱스는 동일
		while(1) {
			//printf("\n");
			while (1) {//pre의 한개 행 * suf 매트릭스
				for(k = i + K, sum = 0;i < k; i++, j += N) {
					sum += mpre[i] * msuf[j];//ret한개 원소값 합산
					//printf("[%f:%f]->[%f]", mpre[i], msuf[j], sum);
				}
				if(rplus) mret[roff] += sum;
				else mret[roff] = sum;
				//printf("%f ", sum);
				if(++roff == n) goto LB1;
				i -= K;//pref의 현 행의 다시 첫번째 컬럼 옵셋 복원
				if(j < suf_n) j = j - suf_sz + 1;//suf의 현 컬럼 옵셋을 첫째 행의 것으로 복원후 다음 컬럼 증가
				else {
					j -= (LAST_OVER_SUF_SZ);//ㄴ.마지막에 증가된 suf의 마지막 컬럼 로우 옵셋 하나 이전으로 복원
					break;
				}
			}
			//printf("\n");
			i += K;//pre의 다음 행
			if(i < pre_n) j -= suf_sz;//suf의 첫번째 행 첫번째 컬럼 복원
			else {//다음 배치
				suf_n = j + suf_sz + LAST_OVER_SUF_SZ;
				pre_n = i + pre_sz;
			}
		}
		break;
	case 1://A^B
		i = pre_n + ((roff % sz) / N);//현 배치옵셋 + 현 배치내의 로우인덱스(pre와 ret는 로우가 일대일 매칭)
		pre_n += M;
		suf_n = j + suf_sz + LAST_OVER_SUF_SZ;
		j += roff % N;//초기 suf컬럼옵셋, suf 배치 옵셋 + ret/suf컬럼 인덱스(roff % N) //ret와 suf의 컬럼인덱스는 동일
		while(1) {
			while(1) {//pre의 한개 행 * suf 매트릭스
				for(k = i + K * M, sum = 0;i < k; i += M, j += N) sum += mpre[i] * msuf[j];//ret한개 원소값 합산
				if(rplus) mret[roff] += sum;
				else mret[roff] = sum;
				if(++roff == n) goto LB1;
				i -= (K * M);//pref의 현 행의 다시 첫번째 컬럼 옵셋 복원
				if(j < suf_n) j = j - suf_sz + 1;//suf의 현 컬럼 옵셋을 첫째 행의 것으로 복원후 다음 컬럼 증가
				else {
					j -= (LAST_OVER_SUF_SZ);//마지막에 증가된 suf의 마지막 컬럼 로우 옵셋 하나 이전으로 복원
					break;
				}
			}
			i++;//ㄴ.pre의 다음 행
			if(i < pre_n) j -= suf_sz;//suf의 첫번째 행 첫번째 컬럼 복원
			else {//다음 배치
				i = (i - M) + pre_sz;//위 ㄴ)에서 이번 배치의 마지막 행까지 증가된 로우 옵셋을 먼저 배치의 처음행으로 복권후 pre사이즈를 더하여 다음 배치의 시작 옵셋 설정.
				suf_n = j + suf_sz + LAST_OVER_SUF_SZ;
				pre_n = i + M;
			}
		}
		break;
	case 2://AB^
		pre_n += pre_sz;//ret의 로우 인덱스는 pre의 로우 인덱스, 배치 전체 범위에서 pre와 ret의 로우 인덱스 일대일 매칭
		i = (roff / N) * K;
		suf_n = j + suf_sz;
		j += (roff % N) * K;//suf 배치 옵셋 + ret컬럼인덱스/suf로우인덱스(roff % N) * suf컬럼사이즈(K) //ret의 컬럼인덱스는 suf의 로우인덱스
		while(1) {
			while(1) {//pre의 한개 행 * suf 매트릭스
				for(k = i + K, sum = 0;i < k; i++, j++) sum += mpre[i] * msuf[j];//ret한개 원소값 합산
				if(rplus) mret[roff] += sum;
				else mret[roff] = sum;
				if(++roff == n) goto LB1;
				i -= K;//pref의 현 행의 다시 첫번째 컬럼 옵셋 복원
				if(j == suf_n) break;//ㄱ.
			}
			i += K;//pre의 다음 행
			if(i < pre_n) j -= suf_sz;//suf의 첫번째 행 첫번째 컬럼 복원
			else {//다음 배치, 위 ㄱ)에서 j가 하나 증가된 상태에서 왔기때문에 현 j는 suf의 다음 배치 첫번째 행 첫번째 컬럼 옵셋
				suf_n = j + suf_sz;
				pre_n = i + pre_sz;
			}
		}
		break;
	case 3://A^B^
		i = pre_n + ((roff % sz) / N);//현 배치옵셋 + 현 배치내의 로우인덱스(pre와 ret는 로우가 일대일 매칭)
		pre_n += M;
		suf_n = j + suf_sz;
		j += (roff % N) * K;//suf 배치 옵셋 + ret컬럼인덱스/suf로우인덱스(roff % N) * suf컬럼사이즈(K) //ret의 컬럼인덱스는 suf의 로우인덱스
		while(1) {
			while(1) {//pre의 한개 행 * suf 매트릭스
				for(k = i + K * M, sum = 0;i < k; i += M, j++) sum += mpre[i] * msuf[j];//ret한개 원소값 합산
				if(rplus) mret[roff] += sum;
				else mret[roff] = sum;
				if(++roff == n) goto LB1;
				i -= (K * M);//pref의 현 행의 다시 첫번째 컬럼 옵셋 복원
				if(j == suf_n) break;//ㄱ
			}
			i++;//ㄴ.pre의 다음 행
			if(i < pre_n) j -= suf_sz;//suf의 첫번째 행 첫번째 컬럼 복원
			else {//다음 배치, 위 ㄱ)에서 j가 하나 증가된 상태에서 왔기때문에 현 j는 suf의 다음 배치 첫번째 행 첫번째 컬럼 옵셋
				i = (i - M) + pre_sz;//위 ㄴ)에서 이번 배치의 마지막 행까지 증가된 로우 옵셋을 먼저 배치의 처음행으로 복권후 pre사이즈를 더하여 다음 배치의 시작 옵셋 설정.
				suf_n = j + suf_sz;
				pre_n = i + M;
			}
		}
		break;
	}
LB1:;
	return n - idx_origin * idx_width;
}
