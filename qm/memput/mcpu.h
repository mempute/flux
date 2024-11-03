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
		oi = roff / outer_sz;//split�� ��ũ ������ ������ �� ���� �޸𸮿��� inner_sz������ �����
		soff = roff % outer_sz;//split�� ��ũ���� ���� �ɼ�
		si = soff / inner_sz;//split�� ���� ���� �̹� ���ҹ�°
		soff = oi * inner_sz + soff % inner_sz;//���ҹ�°���� �ɼ�
		ns = (oi + 1) * inner_sz;
		rsplit_mhost = *((T **)rsplit_mhosts + si);//�̹� ���� �޸�

		for(;roff < n;) {
			if(soff < ns) {//�̹� ���ҳ��̸� �̹� ���ҹ��� ���� �� �ɼ� ����
				if(bw) *(rsplit_mhost + soff) += *(m_split + roff);
				else *(rsplit_mhost + soff) = *(m_split + roff);
				//printf("[%d](%p) %d %d %f %f\n", si, rsplit_mhost, roff, soff, *(rsplit_mhost + soff), *(m_split + roff));
				soff++;
				roff++;
			} else {
				if(++si == nsplit) {
					si = 0;//split�� ��ũ ������ ������ �Ѿ����� ���� �ε��� �ʱ�ȭ
					oi++;
					ns = (oi + 1) * inner_sz;
				}
				soff = oi * inner_sz;
				rsplit_mhost = *((T **)rsplit_mhosts + si);
			}
		}
	} else if(nstep) {//each map, cpu������ �� ���̽��� ������� �����Ƿ� �ʿ������ �׳� �д�.
		intt *sbase = P_LINK_VAR2(intt, pcxt, ccv->sbaseCat);
		intt *sdim = P_LINK_VAR2(intt, pcxt, ccv->sdimCat);
		intt inner_sz = (axis == pdim - 1 ? 1 : MRANK_SIZE(prank, axis + 1));
		nsplit *= -1;
		oi = roff / outer_sz;
		soff = roff % outer_sz;//split�� ��ũ���� ���� �ɼ�
		si = soff / inner_sz;//split�� ���� ���� �̹� ���ҹ�°
		soff = oi * *(sdim + si) + (soff - *(sbase + si));//���ҹ�°���� �ɼ�
		ns = (oi + 1) * *(sdim + si);
		rsplit_mhost = *((T **)rsplit_mhosts + si);//�̹� ���� �޸�

		for(;roff < n;) {
			if(soff < ns) {//�̹� ���ҳ��̸� �̹� ���ҹ��� ���� �� �ɼ� ����
				if(bw) *(rsplit_mhost + soff) += *(m_split + roff);
				else *(rsplit_mhost + soff) = *(m_split + roff);
				soff++;
				roff++;
			} else {
				si += *(sdim + si) / inner_sz;
				if(si == nsplit) {
					si = 0;//split�� ��ũ ������ ������ �Ѿ����� ���� �ε��� �ʱ�ȭ
					oi++;
				}
				if(*((T **)rsplit_mhosts + si) != rsplit_mhost) {
					soff = oi * *(sdim + si);//���� ��Ʈ���������� �ƿ��� oi��°�� ���� �ɼ�, ���� ��Ʈ�������̹Ƿ�
					ns = (oi + 1) * *(sdim + si);//sdim������ �յ��ϰ� sdim������ oi�� �ϳ� �����ѹ�°�� ��谡 �ȴ�.
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
		soff = roff % outer_sz;//split�� ��ũ���� ���� �ɼ�
		for(si = 0;*(sbase + si) + *(sdim + si) <= soff; si++);//split�� ���� ���� �̹� ���ҹ�° find
		soff = oi * *(sdim + si) + (soff - *(sbase + si));//split�� ���� ���� �̹� ���ҹ�°���� �ɼ�
		ns = (oi + 1) * *(sdim + si);
		rsplit_mhost = *((T **)rsplit_mhosts + si);//�̹� ���� �޸�

		for(;roff < n;) {
			if(soff < ns) {//�̹� ���ҳ��̸� �̹� ���ҹ��� ���� �� �ɼ� ����
				if(bw) *(rsplit_mhost + soff) += *(m_split + roff);
				else *(rsplit_mhost + soff) = *(m_split + roff);
				soff++;
				roff++;
			} else {
				if(++si == nsplit) {
					si = 0;//split�� ��ũ ������ ������ �Ѿ����� ���� �ε��� �ʱ�ȭ
					oi++;
				}
				soff = oi * *(sdim + si);//���� ��Ʈ���������� �ƿ��� oi��°�� ���� �ɼ�, ���� ��Ʈ�������̹Ƿ�
				ns = (oi + 1) * *(sdim + si);//sdim������ �յ��ϰ� sdim������ oi�� �ϳ� �����ѹ�°�� ��谡 �ȴ�.
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
	intt outer_sz = MRANK_SIZE(prank, axis);//������ ��Ʈ�������� (�ܸ�����)���� ������� ������
	intt oi, si, soff, ns;

	if(ncat > 0) {
		intt inner_sz = (axis == pdim - 1 ? 1 : MRANK_SIZE(prank, axis + 1)) * nstep;
		oi = roff / outer_sz;
		soff = roff % outer_sz;//split�� ��ũ���� ���� �ɼ�
		si = soff / inner_sz;//split�� ���� ���� �̹� ���ҹ�°
		soff = oi * inner_sz + soff % inner_sz;//���ҹ�°���� �ɼ�
		ns = (oi + 1) * inner_sz;
		pcat_mhost = *((T **)pcat_mhosts + si);//�̹� ���� �޸�

		for(;roff < n;) {
			if(soff < ns) {//�̹� ���ҳ��̸� �̹� ���ҹ��� ���� �� �ɼ� ����
				if(bw) *(m_rcat + roff) += *(pcat_mhost + soff);
				else *(m_rcat + roff) = *(pcat_mhost + soff);
				soff++;
				roff++;
			} else {
				if(++si == ncat) {
					si = 0;//split�� ��ũ ������ ������ �Ѿ����� ���� �ε��� �ʱ�ȭ
					oi++;
					ns = (oi + 1) * inner_sz;
				}
				soff = oi * inner_sz;
				pcat_mhost = *((T **)pcat_mhosts + si);//
			}
			//printf("[%d](%p) %d %d %d %f %f\n", i, pcat_mhost, sdim, roff, cat_off, *(m_rcat + roff), *(pcat_mhost + cat_off));
		}
	} else if(nstep) {//each map, cpu������ �ѹ��� ���ҿɼ¿� ���� ���ҹ�°�� ã�� ���Ĵ� �ݺ��̹Ƿ� ���������� �� ���Һ��� �ɼ���  
		intt *sbase = P_LINK_VAR2(intt, pcxt, ccv->sbaseCat);//����Ͽ� �ӵ��� ������ ������ ũ�� �����Ƿ� �� ���̽��� ������� 
		intt *sdim = P_LINK_VAR2(intt, pcxt, ccv->sdimCat);	//�����Ƿ� �ʿ������ gpu�� ��ì �������� �׳� �д�.
		intt inner_sz = (axis == pdim - 1 ? 1 : MRANK_SIZE(prank, axis + 1));
		ncat *= -1;
		oi = roff / outer_sz;
		soff = roff % outer_sz;//split�� ��ũ���� ���� �ɼ�
		si = soff / inner_sz;//split�� ���� ���� �̹� ���ҹ�°
		soff = oi * *(sdim + si) + (soff - *(sbase + si));//���ҹ�°���� �ɼ�
		ns = (oi + 1) * *(sdim + si);
		pcat_mhost = *((T **)pcat_mhosts + si);//�̹� ���� �޸�

		for(;roff < n;) {
			if(soff < ns) {//�̹� ���ҳ��̸� �̹� ���ҹ��� ���� �� �ɼ� ����
				if(bw) *(m_rcat + roff) += *(pcat_mhost + soff);
				else *(m_rcat + roff) = *(pcat_mhost + soff);
				//printf("[%d][%d](%p) %d %d %f %f\n", *(sdim + si), *(sbase + si), pcat_mhost, roff, soff, *(m_rcat + roff), *(pcat_mhost + soff));
				soff++;
				roff++;
			} else {
				si += *(sdim + si) / inner_sz;
				if(si == ncat) {
					si = 0;//split�� ��ũ ������ ������ �Ѿ����� ���� �ε��� �ʱ�ȭ
					oi++;
				}
				if(*((T **)pcat_mhosts + si) != pcat_mhost) {
					soff = oi * *(sdim + si);//���� ��Ʈ���������� �ƿ��� oi��°�� ���� �ɼ�, ���� ��Ʈ�������̹Ƿ�
					ns = (oi + 1) * *(sdim + si);//sdim������ �յ��ϰ� sdim������ oi�� �ϳ� �����ѹ�°�� ��谡 �ȴ�.
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
		soff = roff % outer_sz;//split�� ��ũ���� ���� �ɼ�
		for(si = 0;*(sbase + si) + *(sdim + si) <= soff; si++);//split�� ���� ���� �̹� ���ҹ�° find
		soff = oi * *(sdim + si) + (soff - *(sbase + si));//split�� ���� ���� �̹� ���ҹ�°���� �ɼ�, ���ڴ� ���� ��Ʈ�������� �ɼ�, ���ڴ� ���� ��Ʈ�������� �����೻�� �̹� ���Ҹ�Ʈ������ �ɼ�
		ns = (oi + 1) * *(sdim + si);
		pcat_mhost = *((T **)pcat_mhosts + si);//�̹� ���� �޸�

		for(;roff < n;) {
			if(soff < ns) {//�̹� ���ҳ��̸� �̹� ���ҹ��� ���� �� �ɼ� ����
				if(bw) *(m_rcat + roff) += *(pcat_mhost + soff);
				else *(m_rcat + roff) = *(pcat_mhost + soff);
				//printf("[%d][%d](%p) %d %d %f %f\n", *(sdim + si), *(sbase + si), pcat_mhost, roff, soff, *(m_rcat + roff), *(pcat_mhost + soff));
				soff++;
				roff++;
			} else {
				if(++si == ncat) {
					si = 0;//split�� ��ũ ������ ������ �Ѿ����� ���� �ε��� �ʱ�ȭ
					oi++;
				}
				soff = oi * *(sdim + si);//���� ��Ʈ���������� �ƿ��� oi��°�� ���� �ɼ�, ���� ��Ʈ�������̹Ƿ�
				ns = (oi + 1) * *(sdim + si);//sdim������ �յ��ϰ� sdim������ oi�� �ϳ� �����ѹ�°�� ��谡 �ȴ�.
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
	for(intt i = 1;i < ndim; i++) {/*[2,1,1,3]�� ��� ��ũ�� [6,-3.-3, 3]�̰� ��ũ�� 1�� ���� �ΰ� ���� ���ӵ� 3�� �ɼ��� �ߺ��ǰ�*/\
		__off += MRANK_SIZE(srank, i) * *(idx + j++);/*���ɰ� ������ idx�� [m,0,0,n]�� ���� ��ũ�� 1�� ������ idx�� 0�� ���ۿ� */\
	}											/*�����Ƿ� ��ũ�� 1�� ������ 0�� ������ off�� �������� 3���� �ߺ��ǰ� �������� �ʴ´�.*/\
	__off += *(idx + j);\
}
#define _lead_offset2idx(nbro, cdim, ndim, srank, _off, cidx, end_check) {\
	if(nbro) {\
		intt _i = 0;\
		for(;_i < cdim - ndim; _i++) cidx[_i] = 0;\
		_offset2idx(ndim, srank, _off, &cidx[_i]);\
	} else cidx[0] = _off;\
	end_check = 0;/*bro_offset�� ��)���� ����üũ�� ���*/\
}
#define _bro_offset(nbro, bro_dim, bro_idx, cdim, crank, cidx, _off, end_check) {\
	intt i = nbro - 1;\
	if(end_check) break;/*��.*/\
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
//original - ��� ��찡 ���Ե� ���� ����
template<typename T>
intt carith_t(void *pcxt, T *m_pari, T *m_sari, T *m_rari, intt r_size, intt idx_origin, intt idx_width, T sval, sytet aop, T rplus)
{//Mul - �������� �� �Ʒ� ������ ��ȸ�ϸ� ä���ֹǷ� �̿� ���� ��ǥ���
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
			for(;;) {//rrank�� �Ѱ� ���ҿ� ���Ͽ� rrank�� ������ 1�� ��ũ���� ���������� ��ȸ�Ͽ� ��ε�ĳ���õ� �͵��� pre�� suf ����
				_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, coff, end_check);
				if(coff < 0) break;
				if(m_pari) {//�����Ŀ��� pref�� �����Ŀ��� ���ϵǴ� ��ε�ĳ��Ʈ�� ��Ʈ������ ������ mrank�� �ɼ°��ǹǷ�
					ppre = m_pari + coff;//�ٷ� ���.
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
{//��ε� �ɽ�Ʈ�� ���� ���� ��Ʈ���� �ϴ��� ���� 
	ArithVar *arv = (ArithVar *)pcxt;
	intt roff = idx_origin * idx_width, coff;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	T rval;//pre�� suf�� ret�� ��ġ�� pre�� suf�� ���Ǳ����� �����Ǳ� ���� ���µǹǷ� ���� �Ի��ϰ� �����ϱ����� 
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
{//��� ������ ���� ��ε� �ɽ�Ʈ(������� 1�� ������ ����)
	ArithVar *arv = (ArithVar *)pcxt;
	intt roff = idx_origin * idx_width, coff;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt osz = arv->zarOut, ssz = arv->zarSuf, psz = arv->zarPre;
	T rval;
	if(arv->bwGetOri) {
		if(osz < psz) {//pre����� out���� ū ���, pre�� �����Ŀ��� ���� ��Ʈ�����̹Ƿ� pre�� �� ���� ���� ����.
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
		} else {//pre�� out�� ����� ���� ���
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
{//������ ����� ��ε� �ɽ�Ʈ(������� 1�� ������ ����)
	ArithVar *arv = (ArithVar *)pcxt;
	intt roff = idx_origin * idx_width;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	T rval;
	if(arv->bwGetOri) {//m_sari�� �����Ķ��� ���
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
		case ABP_MINUS_PREF://������ ����̹Ƿ� pref�� ����.
			break;
		case ABP_MINUS_SUFF:
			for(;roff < n; roff++) {
				if(rplus) *(m_rari + roff) += *(m_sari + roff) * -1;
				else *(m_rari + roff) = *(m_sari + roff) * -1;
			}
			break;
		case ABP_DIV_PREF://������ ����̹Ƿ� pref�� ����.
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
{//������ ����� ��ε� �ɽ�Ʈ(������� 1�� ������ ����)
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
		case ABP_MINUS_SUFF://������ ����̹Ƿ� suff�� ����.
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
		case ABP_DIV_SUFF://������ ����̹Ƿ� suff�� ����.
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
{//������� 1�� ������ �����ϴ� ��ε� �ɽ�Ʈ, ��� ������ ����̸� ����� ������� 1�� ������ �ǹ̰� �������Ƿ� Ÿ�� 2�� ���̽��� �����.
	ArithVar *arv = (ArithVar *)pcxt;
	intt roff = idx_origin * idx_width, poff, soff;
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	intt *mrank = arv->arRankMast, *prank = arv->arRankPre, *srank = arv->arRankSuf, *rrank = arv->arRankRet;
	intt npre = arv->narPre, nsuf = arv->narSuf, nmast = arv->narMast, nret = arv->narRet;
	intt cidx[MX_DIM], tmp_idx[MX_DIM];
	T rval;
	bool end_check;

	if(arv->bwGetOri) {
		if(arv->zarOut < arv->zarPre) {//pre����� out���� ū ���, pre�� �����Ŀ��� ���� ��Ʈ�����̹Ƿ� pre�� �� ���� ���� ����.
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
		} else {//pre�� out�� ����� ���� ���
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
#define _jseq2offset(jseq, jdim, jrank, spr_jo, joff) {/*�� �ε��� ���� �κ��� �ٷ� joff������*/\
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
#define _jseq2offset2(jseq, jdim, jrank, spr_jo, idx, joff) {/*���� ����, �ε��� ���� ����*/\
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
	for(intt i = 1;i < _nout; i++, j++) {/*�ش� �ε����� �ϳ� �Ʒ� ��ũ�� ������� ���� ���̹Ƿ� i�� 1����*/\
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
	sidx[s] = off;/*������ �ε����� suf matrix�� ���� �ǰ� ���� �ɼ��� �ȴ�.*/\
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
//oksz�� ���� ����� �ǰ� ������������ �����޸𸮰� �����ǰ� ��������� ������ �����̰� �����忡 ���� ���ÿ� �����޸𸮷� ����ǹǷ� �����޸� ũ��� �� ������� �����Ѵ�.
//jksz�� sz_smit�� ���� ������� ũ�� oksz�� �ǰ� �ƴϸ� sz_smit�� �����ȴ�.
//jcyc - �ʱⰪ  0���� ����, irecyc - �ʱⰪ nrecyc. nrecycc, ncycj - �Ѵ� �ѹ� �� �ݺ��ҰŸ� 1
#define cache_load(n, roff_ori, roff, okfit, jkfit, oksz, jksz, nrecycc, ncycj, njoint, irecyc, \
	jcyc, tid, bsz, inc, cache_axis, nout, out_rank, tmp_idx, spr_out, spr_jcache, spr_jleaf, njo_cache, \
	jrank_cache, njo_leaf, jrank_leaf, _m_pdot, m_pdot, _m_sdot, m_sdot, m_cache, m_leaf, leaf_idx, \
	cache_bank, lb) {\
	intt sz_smit = MRANK_SIZE(out_rank, cache_axis +1);/*��.pre�� suf�� �ǳ��� ��ġ�� ���� ������(�����ϸ� ���� �ֻ���)*/\
	intt joff, jseq;\
	_offset2idx(nout, out_rank, roff, tmp_idx);\
	for(i = 0, m_pdot = _m_pdot, m_sdot = _m_sdot;i < nout; i++) {\
		if(spr_out[i].rkPref) m_pdot += (tmp_idx[i] * spr_out[i].rksz);\
		else m_sdot += (tmp_idx[i] * spr_out[i].rksz);\
	}/*m_pdot�� m_sdot ������� �ƴ� �� ��)�� ������ �����ϴ� �����Ͱ� ȹ��ȴ�.*/\
	if(spr_out[cache_axis].rkPref) {\
		m_cache = m_pdot;/*�����ϴ� �� ���� �� ���� �޸�(�� ���� �Ѱ����� �����������μ�*/\
		m_leaf = m_sdot;\
	} else {\
		m_cache = m_sdot;/*�ݺ��Ǿ� �ϴ� �޸��� ������ �޸� ������ ȹ��)*/\
		m_leaf = m_pdot;\
	}\
	if(irecyc++ < nrecycc) {/*�����ϴ� �� ���� �� ���� �޸��� ���� ���� ����, �����ϴ� ���ϴܸ�������(sz_smit)*/\
		if(okfit == 0 && irecyc == nrecycc) {/*������������(oksz)���� Ŀ�� ���ҵȰ��μ� ������ ���� ����(ĳ�� ���� ������ ��°��)*/\
			inc = sz_smit % oksz;/*���� �ɼ� ������ �̹� ���� ���� ó���� �͸�ŭ�� ���� ��Ű�� �����Ѵ�.*/\
			if(tid < inc) {\
				/*printf("******* %d %d %d %d \n", tid, ttid, sz_smit, sz_smit % oksz);*/\
			} else {/*���� ���� ������ ��� tid���� �������� �ʴ´�.*/\
				/*printf("++++++ %d %d %d %d \n", tid, ttid, sz_smit, sz_smit % oksz);*/\
				continue;\
			}\
		}\
	} else {/*�����ϴ� �� ���� �� ���������� ���� �Ѱ��� ���� �������� ���ҵ��� ���� �޸𸮿� ����*/\
		/*printf("33 %d %d %d %d\n", jseq, jksz, njoint, sz_smit);*/\
		if(roff >= n) {/*��� ��Ʈ������ ��� ��ȸ������*/\
			if(jcyc < ncycj) {/*���� ���� ���ҵǾ� ���������� ����Ŀ�� ������ ������*/\
				jcyc++;/*������ �� ��� ��Ʈ���� ��ä�� ���Ͽ� ��ȸ �غ�*/\
				roff = roff_ori;\
				if(okfit == 0) __syncthreads();/*��)������ ����*/\
				/*printf("------------\n");*/\
				goto lb;/*ù��° ������ roff�̵����� �ʱ�����*/\
			} else {\
				break;/*���̻� ���ο��� �������� ������ ��ä ����*/\
			}\
		}\
		if(inc != bsz) inc = bsz;/*���� ���� ������� ����ġ*/\
		irecyc = 0;\
		jseq = (tid % jksz) + jcyc * jksz;/*�� tid�� ���° ���� ������ �ش��ϴ��� ���(njoint�� jksz���� �۾Ƶ� ������)*/\
		if(okfit == 0) __syncthreads();/*��.sz_smit�� ���� ������(oksz)�� �Ѿ ���� ������ �̾��� ��� ���� ������ ���������μ� ��������� ������ ���� ��������� ����Ǿʾ����Ƿ� ����ȭ*/\
		if(jseq < njoint) {/*��.�� ���ΰ���(njoint)�� ����Ŀ��(jksz)���� ���ų� njoint�� sz_smit���� Ŀ�� ���ҵǴ� ���*/\
			/*���� �������� jseq�� njoint�̻� �� tid���� �� ���λ���� ��� �͵��̹Ƿ� ĳ���� ����ʰ� �� ���ʸ� �Ѵ�.*/\
			/*printf("44 %d %d\n", jseq, njoint);*/\
			_jseq2offset(jseq, njo_cache, jrank_cache, spr_jcache, joff);\
			/*printf("%d %d %d %d %f\n", jksz, oksz, joff, tid, *(m_cache + joff));*/\
			cache_bank[tid] = *(m_cache + joff);/*�����ϴ� �� ���� ���������(sz_smit) �ɼ��� tid�� ����Ŀ�� ��� ��ġ�� �� ���� ���� ���� ����*/\
		}\
		if(jcyc == ncycj && jkfit == 0) __syncthreads();/*���μ����� ����� sz_smit���� ������ ���ҵ� �������� ���������� sz_smit�� ���� ������ ��)�� ��ŵ�ǹǷ� ����ȭ����, ���� �߰��� �����Ƿ� ����ȭ �������.*/\
	}\
	m_cache = &cache_bank[(tid / sz_smit) * sz_smit];/*�����ϴ� �� ���� �޸�(��Ʈ����)�� �� ���� �޸�(��Ʈ����)�� ĳ���Ǿ� ����Ǵ� ���ν��� ������ ȹ��(�� ���� ����Ŭ ��°��), sz_smit�� ���������� �̻��� ��� ĳ���޸��� 0��°*/\
	jseq = jcyc * jksz;\
	_jseq2offset2(jseq, njo_leaf, jrank_leaf, spr_jleaf, leaf_idx, joff);\
	m_leaf += joff;\
	/*printf("44 %d %d %d %d %f\n", tid, jseq, (tid / sz_smit) * sz_smit, joff, *m_leaf);*/\
}
typedef struct {
	sytet useCublas;
	sytet bwGetOri;
	sytet jdimEqual;
	sytet transOrder;//cublas������ ���.
	bool intervOut, retFirst, fitOutKernel, fitJoKernel;
	intt ndimPre, ndimSuf, ndimRet;
	intt nJointAxis;//joint axis���� dim�� ���� ��
	intt njoPre, njoSuf, njoRet;//����Ʈ�Ǵ� ���� ����
	intt joAxisPre[MX_DIM], joAxisSuf[MX_DIM], joAxisRet[MX_DIM];//����Ʈ�Ǵ� ���� �ε���, Ret�� ��� �����Ķ� ���ε� ���� �ǹ�
	intt joRankPre[MX_DIM], joRankSuf[MX_DIM], joRankRet[MX_DIM];//�������� ���Һ��� �������� ���� ��ũ ������, dot ��� ��翡�� ������� �ȴ�.
	intt noutPre, noutSuf, noutRank, noutRet;//��µǴ� ���� ����
	intt outAxisPre[MX_DIM], outAxisSuf[MX_DIM], outAxisRet[MX_DIM];//��µǴ� ���� �ε���, Ret�� ��� �����Ķ� ��� ���� �ǹ�
	intt outRankPre[MX_DIM], outRankSuf[MX_DIM];//����Ʈ�� �ƴ� ���� ��ũ ������, dot ��� shape�� �ȴ�.
	intt szRankPre[MX_DIM], szRankSuf[MX_DIM], szRankRet[MX_DIM];//���� ��Ʈ���� ������ dot�Ǳ� ���� ������ ���� ��ũ ������
	intt outAxis[MX_DIM], outRank[MX_DIM];
	SparseRank sprPreOut[MX_DIM], sprPreJo[MX_DIM], sprSufOut[MX_DIM], sprSufJo[MX_DIM];
	intt szSuf, szJoKernel, szOutKernel, nrecycCache, ncycJo, axisCache;
	intt shareUnit;//���� �޸� ���� ���������� ���.
	intt prem, sufn, joik, lda, ldb, ldc;//cublas������ ���.
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
	//�� roff�� ret��Ʈ������ ������ �ɼ��̰� �̰��� �̹� ��Ʈ���� ���� ���� out axis rank�������� ��ȯ�Ѵ�.
	_offset2idx(nout, out_rank, roff, ret_idx);
	for(i = 0;i < nout; i++) {
		if(spr_out[i].rkPref) m_pdot += (ret_idx[i] * spr_out[i].rksz);
		else m_sdot += (ret_idx[i] * spr_out[i].rksz);
	}
	if(jdim_equal) {//���� ���� ��ũ�� ������ �ѹ��� �ʱ�ȭ
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
		for(sum = 0;;) {//���������� ���, ���� �ε����� ���������� �����ϸ� ���� ���� ����, ����Ʈ ���� �ε��� ����
			//printf("%f %f\n", *m_pdot, *m_sdot);
			sum += *m_pdot * *m_sdot;
			if(jdim_equal) {//���� ���� �ε����� ������ ��ǥ�� pre join�ε����� ����
				for(;;) {//pre �������� �ε��� ����
					if(pjdim == ++i_pj) {//����(����) �ø��� ���ÿ� ���������� ���� ��� �ʱ�ȭ�� �ڵ����� �ȴ�.
						i_pj = 0;
						i = njo_pre2;
J0:;					m_pdot -= spr_pre_jo[i].rktsz;
						m_sdot -= spr_suf_jo[i].rktsz;
						if(--i < 0) goto LB1;
						else {//�߰� ���� ����
							if(spr_pre_jo[i].rkdim == ++pj_idx[i]) {
								pj_idx[i] = 0;
								goto J0;
							} else {
								m_pdot += spr_pre_jo[i].rksz;//���� ���� ����(����)�� ����
								m_sdot += spr_suf_jo[i].rksz;
								break;
							}
						}
					} else {//���� ���� ����
						m_pdot += pjsz;//�� ����(����)�� ����
						m_sdot += sjsz;
						break;
					}
				}
			} else {
				for(;;) {//pre �������� �ε��� ����
					if(pjdim == ++i_pj) {//����(����) �ø��� ���ÿ� ���������� ���� ��� �ʱ�ȭ�� �ڵ����� �ȴ�.
						i_pj = 0;
						i = njo_pre2;
J1:;					m_pdot -= spr_pre_jo[i].rktsz;
						if(--i < 0) break;
						else {
							if(spr_pre_jo[i].rkdim == ++pj_idx[i]) {
								pj_idx[i] = 0;
								goto J1;
							} else {
								m_pdot += spr_pre_jo[i].rksz;//���� ���� ����(����)�� ����
								break;
							}
						}
					} else {
						m_pdot += pjsz;//�� ����(����)�� ����
						break;
					}
				}
				for(;;) {//suf �������� �ε��� ����
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
								m_sdot += spr_suf_jo[i].rksz;//���� ���� ����(����)�� ����
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
		for(;;) {//��� ���� �ε��� ����(pre�� suf�� ��� �� �����Ͽ�)
			if(podim == ++i_po) {//podim�� pre�� suf�� ���� �������� �� ������ ��ġ�ϴ� ���� �����, i_po, posz, lastout_is_pref�� ��������)
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
						if(spr_out[i].rkPref) m_pdot += spr_out[i].rksz;//���� ���� ����(����)�� ����
						else m_sdot += spr_out[i].rksz;
						break;//suf out �߰� ���� ���� ����
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
			if(tdim == ++it) {//tdim�� �� ������ ��ġ�ϴ� ���� �����, it, tsz, lastout_is_pref�� ��������)
				it = 0;
				i = ndims;
LB1:			m_strs -= tmap[i].trstsz;
				if(--i < 0) goto LB2;
				else {
					if(tmap[i].trsdim == ++ridx[i]) {
						ridx[i] = 0;
						goto LB1;
					} else {
						 m_strs += tmap[i].trssz;//���� ���� ����(����)�� ����
						break;//�߰� ���� ���� ����
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
	if(mean) {//��Ʈ���� ��ü ���� ���� ���
		intt roff = 0;
		for(;roff < r_size; roff++) {
			m_rmet[0] += ((m_smet[roff] - m_tmet[roff]) * (m_smet[roff] - m_tmet[roff]));
		}
		m_rmet[0] /= r_size;
		return 1;
	} else {//��ġ���� ���� ���������� ���
		intt roff = idx_origin * idx_width;
		intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
		for(;roff < n; roff++) {
			m_rmet[roff] = (m_smet[roff] - m_tmet[roff]) * (m_smet[roff] - m_tmet[roff]);
		}
		return n;
	}
}
template<typename T>
intt csum_f(void *pcxt, T *m_smet, T *m_rmet, intt r_size, T *cmul, bool mean) //��ü ��, ��� ���
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
	intt nrkPre, nrkSuf, nrkOut, ovpad;//ovpad���ּ� 8����Ʈ �����ϱ����� 
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
				for(;roff < n; roff++) {//roff�� mret�ɼ�
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
		for(beg = roff;roff < n; roff++) mret[roff / sum_sz] += mpre[roff];//roff�� mpre�ɼ�
		if(ovar->idxOne[0]) {//��� ���ϱ�, �̹� mpre�� ���۰� �� ������ sum_sz(��ġ���� �̿� ������ ��)������ 
			intt n2 = n / sum_sz;//�پ�� ��ġ ����� ����� mret�� ���۰� �� �ɼ����� ��ȯ�Ͽ� �� ������ ��հ� ����
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
		mm[roff] += (1.0f - beta1) * (mg[roff] - mm[roff]);//����: mm[roff] + (l - beta1)*mg[roff] - mm[roff] + mm[roff]*beta1
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
{//mpre�� �����Ķ� ���, msuf�� ���� ����
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
	case DJUST_COPY://mpre�� ���� ����, �ܼ��� �����Ķ� ���⸦ ���ϱ����� ���.
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
	case DJUST_COPY2://mpre�� ���� ����, mpre, msuf�� ���� 1��
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
		for(; roff < n; roff++) {//msuf�� ��ǥ��
			mret[roff] += ((mpre[roff] - msuf[roff]) / *(T *)ovar->idxOne);//��ġ������� ����
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
		//roff�� ��Ʈ���̵忡 ���� Ȯ�� �Ļ��� ��Ʈ������ �ɼ��̹Ƿ� �̷μ� �ε����� ȹ���ϰ� �̷κ��� �ҽ��� �ɼ��� ����Ѵ�.
		ibatch = roff / sz_derive;//��Ʈ���̵忡 ���� �Ļ��� ��Ʈ���������� �̹� �ɼ��� ���� ��ġ�ο� �ε���
		rest = roff - (ibatch * sz_derive);//�Ļ��� ��Ʈ������ �̹� ���� ��ġ�ο��� ���� ������
		//��Ʈ���̵忡 ���� Ȯ��� �Ļ� ��Ʈ�������� �� �ɼ��� �Ļ� ��Ʈ�����󿡼��� ��ġ�ε����� �����̵��� ���� �𼭸���
		//���������� �Ͽ� ��ġ�������� �ο�� �÷� �ε����� ��ȯ�Ѵ�.
		if(outy != 1) {//2d
			irow = rest / sz_derive_row;//�Ļ���Ʈ������ ���� �ϳ��� ��ġ���� �ɼ��� �Ļ���Ʈ���� �� ���� ������� ����
			rest -= irow * sz_derive_row;//�Ļ���Ʈ������ �� ��ġ�� �ο� �ε����� ���ϰ� �� �ο��� ���� �ɼ����κ��� ����
			d2 = 1;						//����� ���Ѵ�.
		}
		else {
			irow = 0;//1d	
			d2 = 0;
		}
		icol = rest / sz_slide;//�Ļ���Ʈ������ ���� ������κ��� �����̵� ��������� �÷� �ε����� ���
		rest -= icol * sz_slide;

		prex *= sz_feat;
		stridey *= prex;//�ο� ��Ʈ���̵带 mpre��Ʈ�������� ������ ��ȯ
		stridex *= sz_feat;//�÷� ��Ʈ���̵带 mpre��Ʈ�������� ������ ��ȯ
		outy *= stridey;//�Ѱ谪�� mpre��Ʈ�������� ������ ��ȯ, �����е� �ȴٸ� prey�� ��Ʈ���̵尡 �ѹ� �� ������ ���� �ϼ��ִ�. 
		outx *= stridex;//�Ѱ谪�� mpre��Ʈ�������� ������ ��ȯ, �����е� �ȴٸ� prex�� ��Ʈ���̵尡 �ѹ� �� ������ ���� �ϼ��ִ�.
		irow *= stridey;//�Ѱ谪�� mpre��Ʈ�������� �������� ��ȯ�����Ƿ� �ʱⰪ�� ��ȯ
		icol *= stridex;//�Ѱ谪�� mpre��Ʈ�������� �������� ��ȯ�����Ƿ� �ʱⰪ�� ��ȯ


		if(slidey != 1) {//2d
			i = rest / slidex;//�Ļ���Ʈ������ �����̵峻�� �ο� �ε��� ��� 
			j = rest % slidex;//�Ļ���Ʈ������ �����̵峻�� �÷� �ε��� ���
		}
		else {//1d
			i = 0;
			j = rest;
		}
		slidey *= prex;//�Ѱ谪�� mpre��Ʈ�������� ������ ��ȯ
		i *= prex;//�Ѱ谪�� mpre��Ʈ�������� �������� ��ȯ�����Ƿ� �ʱⰪ�� ��ȯ(�����̵峻�� �ο��
		j *= sz_feat;	//mpre������ �� ������ ������ ����)
		prey *= prex;//1�� ��ġ������
		for(;; ibatch++) {
			pbatch = mpre + ibatch * prey;//mpre��Ʈ�������� �̹� ��ġ ���� ������ ���.
			for(; irow < outy; irow += stridey) {//�Ļ��ο��ε����� �ҽ����� �ο� ��Ʈ���̵� ������ ������
				py = pbatch + d2 * irow;//d2�� 1d�̸� 0, mpre��Ʈ�������� �ο������� ����.
				for(; icol < outx; icol += stridex) {//�Ļ��÷��ε����� �ҽ����� �÷� ��Ʈ���̵� ������ ������
					pslide = py + icol;//�����̵� ������(���� ����) ������ ����
					for(; i < slidey; i += prex) {//�����̵峻�� �ο� �ϳ��� ����, �����̵峻�� �ο��
						px = pslide + i;//mpre������ �� ������ ������ �����ؾ��Ѵ�.
						for(; j < slidex; j++) {//�����̵峻�� �÷� �ϳ��� ����
							if(roff < n) mret[roff++] = (irow + i >= prey || icol + j >= prex ? 0 : *(px + j));//�����е�
							else return n - idx_origin * idx_width;
							//printf("%f\n", mret[roff - 1]);
						}
						j = 0;
					}
					i = 0;
				}
				icol = 0;//�Ļ���Ʈ���� ���� �ο���ʹ� �÷��� 0���� ����
			}
			irow = 0;//�Ļ���Ʈ���� ���� ��ġ���ʹ� �ο츦 0���� ����
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
	case ACTF_PRELU://msuf�� prelu ����ġ ������
		for(;roff < n; roff++) {
			mret[roff] = mpre[roff] > 0.0 ? mpre[roff] : msuf[roff] * mpre[roff];// x > 0.0f ? x : a * x
		}
		break;
	case DACTF_PRELU://mpre�� ���� ����, msuf�� �����Ķ� ���, mret�� �����Ķ� �Է� ����, 
	{				//m1 - prelu ����ġ ������, m2 - prelu ����ġ ����
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

	if(bw) {//msuf - input, mret - lookup table, mpre - embeded, roff�� mpre ����
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
	} else {//msuf - input, mret - embeded, mpre - lookup table, roff�� mret ����
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
			if(tdim == ++it) {//tdim�� �� ������ ��ġ�ϴ� ���� �����, it, tsz�� ��������)
				it = 0;
				i = ndims;
LB1:			cmem -= slicer[i].sltsz;
				if(--i < 0) goto LB2;
				else {
					if(slicer[i].sldim == ++ridx[i]) {
						ridx[i] = 0;
						goto LB1;
					} else {
						cmem += slicer[i].slsz;//���� ���� ����(����)�� ����
						break;//�߰� ���� ���� ����
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
		poff = (roff / inner_sz) * outer_sz + roff % inner_sz;//roff�� poff���� max���� �������� ��ҵ� �ɼ��̹Ƿ�
		//roff�� inner_sz�� ���� �ε����� poff�������� outer_sz�� ���� ������ ����. ���� �� �ε����� outer_sz�� ���ؼ�
		//����� outer_sz���� poff�� ���ϰ� ���⿡ inner_sz���� �ɼ��� ���� ���� �ɼ� poff�� ȹ���Ѵ�.
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
	//vmax�ܰ迡���� �б�� �� �����Ƿ� rplus�� ����
	if(bw) {//mpre�� arg max index map, msuf�� ������ ����, �Ѵ� poff(��� �ɼ�)���� ����, mret�� �ҽ�, roff�� Ǯ�� �ҽ� �ɼ�
		for(; roff < n; roff++) {
			poff = (roff / outer_sz) * inner_sz + roff % inner_sz;//�ɼ� ���
			if(*(mpre + poff) == (roff % outer_sz) / inner_sz) *(mret + roff) = *(msuf + poff);
			else *(mret + roff) = 0;
			//printf("%d %d %f %f %f\n", poff, roff, *(mpre + poff), *(msuf + poff), *(mret + roff));
		}
	} else {//mpre�� �ҽ�, msuf�� arg max index map, mret�� max ���, roff�� Ǯ�� ��� �ɼ�, poff�� Ǯ�� �ҽ� Ȯ�� �ɼ�
		for(; roff < n; roff++) {
			poff = (roff / inner_sz) * outer_sz + roff % inner_sz;//roff�� poff���� max���� �������� ��ҵ� �ɼ��̹Ƿ�
			//roff�� inner_sz�� ���� �ε����� poff�������� outer_sz�� ���� ������ ����. ���� �� �ε����� outer_sz�� ���ؼ�
			//����� outer_sz���� poff�� ���ϰ� ���⿡ inner_sz���� �ɼ��� ���� ���� �ɼ� poff�� ȹ���Ѵ�.
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
		return (n - idx_origin * idx_width) * -1;//roff, r_size�� mret�� �ƴ� mpre�� ���̹Ƿ� �����Ͽ� ��Ʈ���� ��ü �����ϰ� �Ѵ�.
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
		random_device rd;//�ü������ �ð�, ���μ��� ���̵�, ȯ�溯��, �����̸�, ��Ǫ�� �̸����� �ؽ��Ͽ� SHA-1�Լ��� �õ����.
		mt19937 mt(seed < 0 ? rd() : seed);
		//mt19937 mt(1729);//1729��� �õ尪���� ���� ����.
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
		/*for(i = roff;i < n; i++) g_mz[i] = mi[i] * ga[i % dsz];//���� ���谪�� �������� ���Ͽ� ������ ��� ǥ�ذ��� ���� ����
		for(i = roff;i < n; i++) var[i / dsz] += -0.5 * g_mz[i] * md[i];//ǥ�ذ� ���迡 ������ �Է°� ��������� ���� ��
		for(i = roff;i < m; i++) var[i] *= (mv[i] * mv[i] * mv[i]);//�� �տ� �Է°� ǥ������ ������ ���Ͽ� �л� ���� ����
		for(i = roff;i < n; i++) mdmean[i / dsz] += md[i];//�Է°� ��� ���� ��
		for(i = roff;i < m; i++) mdmean[i] /= dsz;//md mean, �Է°� ��� ���� �� ���
		for(i = roff;i < n; i++) mean[i / dsz] += -1.0 * g_mz[i] * mv[i / dsz];//������ ��� ǥ�ذ��� ���迡 ǥ������ ������ ���� ��
		for(i = roff;i < m; i++) mean[i] += -2.0 * var[i] * mdmean[i];//�� �տ� �л� ����� ������� ����� ���� ���� �հ��Ͽ� ��� ���� ����
		for(i = roff;i < n; i++) mr[i] = g_mz[i] * mv[i / dsz] + (2.0 / dsz) * var[i / dsz] * md[i] + (1.0 / dsz) * mean[i / dsz];
		for(i = roff;i < n; i++) {
			g_gm[i % dsz] += mi[i] * mz[i];//���� ���谪�� ������ ��� ǥ�ذ��� ���� ���� ���Ͽ� ���� ���� ����
			g_be[i % dsz] += mi[i];//���� ���谪�� ���Ͽ� ��Ÿ ���� ����
		}*/
		for(i = roff;i < n; i++) {
			g_mz[i] = mi[i] * ga[i % dsz];//���� ���谪�� �������� ���Ͽ� ������ ��� ǥ�ذ��� ���� ����
			q = i / dsz;
			var[q] += -0.5 * g_mz[i] * md[i];//ǥ�ذ� ���迡 ������ �Է°� ��������� ���� ��
			if((i + 1) % dsz == 0) var[q] *= (mv[q] * mv[q] * mv[q]);//�� �տ� �Է°� ǥ������ ������ ���Ͽ� �л� ���� ����
		}
		for(i = roff;i < n; i++) {
			q = i / dsz;
			mdmean[q] += md[i];//�Է°� ��� ���� ��
			if((i + 1) % dsz == 0) mdmean[q] /= dsz;//md mean, �Է°� ��� ���� �� ���
		}
		for(i = roff;i < n; i++) {
			q = i / dsz;
			mean[q] += -1.0 * g_mz[i] * mv[q];//������ ��� ǥ�ذ��� ���迡 ǥ������ ������ ���� ��
			if((i + 1) % dsz == 0) mean[q] += -2.0 * var[q] * mdmean[q];//�� �տ� �л� ����� ������� ����� ���� ���� �հ��Ͽ� ��� ���� ����
		}
		for(i = roff;i < n; i++) {
			q = i / dsz;
			mr[i] = g_mz[i] * mv[q] + (2.0 / dsz) * var[q] * md[i] + (1.0 / dsz) * mean[q];
			r = i % dsz;
			g_gm[r] += mi[i] * mz[i];//���� ���谪�� ������ ��� ǥ�ذ��� ���� ���� ���Ͽ� ���� ���� ����
			g_be[r] += mi[i];//���� ���谪�� ���Ͽ� ��Ÿ ���� ����
		}
	} else {
		/*for(i = roff;i < n; i++) mean[i / dsz] += mi[i];//�Է°� �հ�
		for(i = roff;i < m; i++) mean[i] /= dsz;//�Է°� ���
		for(i = roff;i < n; i++) {
			md[i] = v = (mi[i] - mean[i / dsz]);//��� ����
			mv[i / dsz] += v * v;//�л�, ������� ���� ��
		}
		for(i = roff;i < m; i++) mv[i] = 1.0 / std::sqrt(mv[i] + 1e-9);//ǥ������ ����
		for(i = roff;i < n; i++) mz[i] = md[i] * mv[i / dsz];//�Է°� ��� ������ ǥ������ ������ ���Ͽ� ǥ�ذ� ����
		for(i = roff;i < n; i++) mr[i] = mz[i] * ga[i % dsz] + be[i % dsz];//ǥ�ذ��� ������ ���ϰ� ��Ÿ�� ���Ͽ� ǥ�� ��°� ����.*/
		for(i = roff;i < n; i++) {
			q = i / dsz;
			mean[q] += mi[i];//�Է°� �հ�
			if((i + 1) % dsz == 0) mean[q] /= dsz;//�Է°� ���
		}
		for(i = roff;i < n; i++) {
			q = i / dsz;
			md[i] = v = (mi[i] - mean[q]);//��� ����
			mv[q] += v * v;//�л�, ������� ���� ��
			if((i + 1) % dsz == 0) mv[q] = 1.0 / std::sqrt(mv[q] / dsz + 1e-9);//ǥ������(�л����) ����
		}
		for(i = roff;i < n; i++) {
			mz[i] = md[i] * mv[i / dsz];//�Է°� ��� ������ ǥ������ ������ ���Ͽ� ǥ�ذ� ����
			r = i % dsz;
			mr[i] = mz[i] * ga[r] + be[r];//ǥ�ذ��� ������ ���ϰ� ��Ÿ�� ���Ͽ� ǥ�� ��°� ����.
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
	intt sz = M * N;//��ġ�� ��� K, N�� ��ġ�� ���� �ε����̰� suf��Ʈ������ ��ġ���� ���� ������ ��Ʈ����(N, K)
	intt pre_sz = M * K;//�Ѱ� ��ġ�� pre ��Ʈ���� ������
	intt suf_sz = K * N;//�Ѱ� ��ġ�� suf ��Ʈ���� ������
	intt pre_n = (roff / sz) * pre_sz;//pre ��ġ �ɼ�, ��ġ�ε���(roff / sz) * pre ��Ʈ���� ������
	intt j = (roff / sz) * suf_sz;//suf ��ġ �ɼ�, ��ġ�ε���(roff / sz) * suf ��Ʈ���� ������
	intt suf_n, i, k;
	DT sum;
	switch(T) {
	case 0://AB
		pre_n += pre_sz;
		i = (roff / N) * K;//ret�� �ο� �ε����� pre�� �ο� �ε���, ��ġ ��ü �������� pre�� ret�� �ο� �ε��� �ϴ��� ��Ī
		suf_n = j + suf_sz + LAST_OVER_SUF_SZ;
		j += roff % N;//�ʱ� suf�÷��ɼ�, suf ��ġ �ɼ� + ret/suf�÷� �ε���(roff % N) //ret�� suf�� �÷��ε����� ����
		while(1) {
			//printf("\n");
			while (1) {//pre�� �Ѱ� �� * suf ��Ʈ����
				for(k = i + K, sum = 0;i < k; i++, j += N) {
					sum += mpre[i] * msuf[j];//ret�Ѱ� ���Ұ� �ջ�
					//printf("[%f:%f]->[%f]", mpre[i], msuf[j], sum);
				}
				if(rplus) mret[roff] += sum;
				else mret[roff] = sum;
				//printf("%f ", sum);
				if(++roff == n) goto LB1;
				i -= K;//pref�� �� ���� �ٽ� ù��° �÷� �ɼ� ����
				if(j < suf_n) j = j - suf_sz + 1;//suf�� �� �÷� �ɼ��� ù° ���� ������ ������ ���� �÷� ����
				else {
					j -= (LAST_OVER_SUF_SZ);//��.�������� ������ suf�� ������ �÷� �ο� �ɼ� �ϳ� �������� ����
					break;
				}
			}
			//printf("\n");
			i += K;//pre�� ���� ��
			if(i < pre_n) j -= suf_sz;//suf�� ù��° �� ù��° �÷� ����
			else {//���� ��ġ
				suf_n = j + suf_sz + LAST_OVER_SUF_SZ;
				pre_n = i + pre_sz;
			}
		}
		break;
	case 1://A^B
		i = pre_n + ((roff % sz) / N);//�� ��ġ�ɼ� + �� ��ġ���� �ο��ε���(pre�� ret�� �ο찡 �ϴ��� ��Ī)
		pre_n += M;
		suf_n = j + suf_sz + LAST_OVER_SUF_SZ;
		j += roff % N;//�ʱ� suf�÷��ɼ�, suf ��ġ �ɼ� + ret/suf�÷� �ε���(roff % N) //ret�� suf�� �÷��ε����� ����
		while(1) {
			while(1) {//pre�� �Ѱ� �� * suf ��Ʈ����
				for(k = i + K * M, sum = 0;i < k; i += M, j += N) sum += mpre[i] * msuf[j];//ret�Ѱ� ���Ұ� �ջ�
				if(rplus) mret[roff] += sum;
				else mret[roff] = sum;
				if(++roff == n) goto LB1;
				i -= (K * M);//pref�� �� ���� �ٽ� ù��° �÷� �ɼ� ����
				if(j < suf_n) j = j - suf_sz + 1;//suf�� �� �÷� �ɼ��� ù° ���� ������ ������ ���� �÷� ����
				else {
					j -= (LAST_OVER_SUF_SZ);//�������� ������ suf�� ������ �÷� �ο� �ɼ� �ϳ� �������� ����
					break;
				}
			}
			i++;//��.pre�� ���� ��
			if(i < pre_n) j -= suf_sz;//suf�� ù��° �� ù��° �÷� ����
			else {//���� ��ġ
				i = (i - M) + pre_sz;//�� ��)���� �̹� ��ġ�� ������ ����� ������ �ο� �ɼ��� ���� ��ġ�� ó�������� ������ pre����� ���Ͽ� ���� ��ġ�� ���� �ɼ� ����.
				suf_n = j + suf_sz + LAST_OVER_SUF_SZ;
				pre_n = i + M;
			}
		}
		break;
	case 2://AB^
		pre_n += pre_sz;//ret�� �ο� �ε����� pre�� �ο� �ε���, ��ġ ��ü �������� pre�� ret�� �ο� �ε��� �ϴ��� ��Ī
		i = (roff / N) * K;
		suf_n = j + suf_sz;
		j += (roff % N) * K;//suf ��ġ �ɼ� + ret�÷��ε���/suf�ο��ε���(roff % N) * suf�÷�������(K) //ret�� �÷��ε����� suf�� �ο��ε���
		while(1) {
			while(1) {//pre�� �Ѱ� �� * suf ��Ʈ����
				for(k = i + K, sum = 0;i < k; i++, j++) sum += mpre[i] * msuf[j];//ret�Ѱ� ���Ұ� �ջ�
				if(rplus) mret[roff] += sum;
				else mret[roff] = sum;
				if(++roff == n) goto LB1;
				i -= K;//pref�� �� ���� �ٽ� ù��° �÷� �ɼ� ����
				if(j == suf_n) break;//��.
			}
			i += K;//pre�� ���� ��
			if(i < pre_n) j -= suf_sz;//suf�� ù��° �� ù��° �÷� ����
			else {//���� ��ġ, �� ��)���� j�� �ϳ� ������ ���¿��� �Ա⶧���� �� j�� suf�� ���� ��ġ ù��° �� ù��° �÷� �ɼ�
				suf_n = j + suf_sz;
				pre_n = i + pre_sz;
			}
		}
		break;
	case 3://A^B^
		i = pre_n + ((roff % sz) / N);//�� ��ġ�ɼ� + �� ��ġ���� �ο��ε���(pre�� ret�� �ο찡 �ϴ��� ��Ī)
		pre_n += M;
		suf_n = j + suf_sz;
		j += (roff % N) * K;//suf ��ġ �ɼ� + ret�÷��ε���/suf�ο��ε���(roff % N) * suf�÷�������(K) //ret�� �÷��ε����� suf�� �ο��ε���
		while(1) {
			while(1) {//pre�� �Ѱ� �� * suf ��Ʈ����
				for(k = i + K * M, sum = 0;i < k; i += M, j++) sum += mpre[i] * msuf[j];//ret�Ѱ� ���Ұ� �ջ�
				if(rplus) mret[roff] += sum;
				else mret[roff] = sum;
				if(++roff == n) goto LB1;
				i -= (K * M);//pref�� �� ���� �ٽ� ù��° �÷� �ɼ� ����
				if(j == suf_n) break;//��
			}
			i++;//��.pre�� ���� ��
			if(i < pre_n) j -= suf_sz;//suf�� ù��° �� ù��° �÷� ����
			else {//���� ��ġ, �� ��)���� j�� �ϳ� ������ ���¿��� �Ա⶧���� �� j�� suf�� ���� ��ġ ù��° �� ù��° �÷� �ɼ�
				i = (i - M) + pre_sz;//�� ��)���� �̹� ��ġ�� ������ ����� ������ �ο� �ɼ��� ���� ��ġ�� ó�������� ������ pre����� ���Ͽ� ���� ��ġ�� ���� �ɼ� ����.
				suf_n = j + suf_sz;
				pre_n = i + M;
			}
		}
		break;
	}
LB1:;
	return n - idx_origin * idx_width;
}