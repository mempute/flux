
#include "mgpu.h"
#include "matrix.h"
#include <stdio.h>

__device__ intt didx2offset(intt ndim, intt *srank, intt *idx)
{
	intt off = 0, j = 0;

	for(intt i = 1;i < ndim; i++) {
		off += MRANK_SIZE(srank, i) * *(idx + j++);
	}
	off += *(idx + j);

	return off;
}
__device__ void doffset2idx2(intt out_axis[], intt n_preout_axis, intt rdim, intt rrank[], intt off, intt pidx[], intt sidx[])
{
	intt j = 0, k;

	for(intt i = 1;i < rdim; i++, j++) {//해당 인덱스는 하나 아래 랭크의 사이즈로 나눈 몫이므로 i는 1부터
		k = out_axis[j];
		if(j < n_preout_axis) {
			if(rrank[j] < 0) pidx[k] = 0;
			else {
				pidx[k] = off / MRANK_SIZE(rrank, i);
				off %= MRANK_SIZE(rrank, i);
			}
		} else {
			if(rrank[j] < 0) sidx[k] = 0;
			else {
				sidx[k] = off / MRANK_SIZE(rrank, i);
				off %= MRANK_SIZE(rrank, i);
			}
		}
	}
	sidx[out_axis[j]] = off;//마지막 인덱스는 suf matrix의 것이 되고 남은 옵셋이 된다.
}
__device__ intt dsparse_idx2offset(intt ndim, intt *srank, intt *idx, intt *axis)
{
	intt off = 0, j = 0;

	for(intt i = 1;i < ndim; i++) {
		off += MRANK_SIZE(srank, i) * *(idx + *(axis + j++));
	}
	off += *(idx + *(axis + j));

	return off;
}
__device__ void doffset2idx(intt ndim, intt *srank, intt off, intt *idx)
{
	intt j = 0;

	for(intt i = 1;i < ndim; i++) {
		if(*(srank + j) < 0) *(idx + j++) = 0;
		else {
			*(idx + j++) = off / MRANK_SIZE(srank, i);
			off %= MRANK_SIZE(srank, i);
		}
	}
	*(idx + j) = off;
}
__device__ intt dmoff2soff(intt mdim, intt *mrank, intt sdim, intt *srank, intt moff, intt *idx)
{
	intt i = mdim - 1, j = sdim - 1;

	doffset2idx(mdim, mrank, moff, idx);
	for(;j >= 0; i--, j--) {
		if(*(srank + j) < 0) *(idx + i) = 0;
	}
	return didx2offset(sdim, srank, idx + ++i);
}
__device__ void dlead_offset2idx(intt nbro, intt cdim, intt ndim, intt *srank, intt off, intt cidx[])
{
	if(nbro) {
		intt i = 0;
		for(;i < cdim - ndim; i++) cidx[i] = 0;
		doffset2idx(ndim, srank, off, &cidx[i]);
	} else cidx[0] = off;

	cidx[MX_DIM - 1] = 0;//bro_offset의 ㄱ)에서 종료체크로 사용
}
__device__ intt dbro_offset(intt nbro, intt bro_dim[], intt bro_idx[], intt cdim, intt *crank, intt cidx[])
{
	intt i = nbro - 1, off;

	if(cidx[MX_DIM - 1] == 1) return -1;//ㄱ.

	if(nbro) {
		off = didx2offset(cdim, crank, cidx);

		for(;i >= 0; i--) {
			if(++cidx[bro_idx[i]] == bro_dim[i]) cidx[bro_idx[i]] = 0;
			else break;
		}
		if(i < 0) cidx[MX_DIM - 1] = 1;
		return off;
	} else {
		cidx[MX_DIM - 1] = 1;
		return cidx[0];
	}
}
/*__device__ intt count_over_axis(intt idx[], intt axis)
{
	if(axis == 0) return -1;

	intt n = 1;
	for(intt i = 0;i < axis; i++) n *= idx[i];
	return n;
}*/
template<typename T>
__global__ void ksplit_f(void *pcxt, T *m_split, intt pdim, intt sdim, intt idx_origin, intt idx_width,
	intt nsplit, intt nstep, intt axis, bool bw, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	intt *prank;
	T *rsplit_mdev, *rsplit_mdevs;
	ConcatVar *ccv = (ConcatVar *)pcxt;

	prank = P_LINK_VAR2(intt, pcxt, ccv->szRankPrimary);
	rsplit_mdevs = P_LINK_VAR2(T, pcxt, ccv->mptrDevSecondary);
	intt outer_sz = MRANK_SIZE(prank, axis), inner_sz = (axis == pdim - 1 ? 1 : MRANK_SIZE(prank, axis + 1)) * nstep;
	intt si, soff;

	soff = roff % outer_sz;//split축 랭크내의 분할 옵셋
	si = soff / inner_sz;//split축 분할 단위 이번 분할번째
	soff = (roff / outer_sz) * inner_sz + soff % inner_sz;//분할번째내의 옵셋
	rsplit_mdev = *((T **)rsplit_mdevs + si);//이번 분할 메모리

	if(bw) *(rsplit_mdev + soff) += *(m_split + roff);
	else *(rsplit_mdev + soff) = *(m_split + roff);
	//printf("(%p) %d %d %f %f\n", rsplit_mdev, roff, soff, *(rsplit_mdev + soff), *(m_split + roff));
}

template<typename T>
intt gsplit_t(void *pcxt, T *m_split, intt pdim, intt sdim, intt rsize, intt idx_origin, intt idx_width,
	intt nsplit, intt nstep, intt axis, bool bw)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (rsize > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rsize);

	ksplit_f<T> << <grid, block >> > (pcxt, m_split, pdim, sdim, idx_origin, idx_width, nsplit, nstep, axis, bw, n);
	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gsplit_f(void *pcxt, floatt *m_split, intt pdim, intt sdim, intt rsize, intt idx_origin, intt idx_width,
	intt nsplit, intt nstep, intt axis, bool bw)
{
	return gsplit_t<floatt>(pcxt, m_split, pdim, sdim, rsize, idx_origin, idx_width, nsplit, nstep, axis, bw);
}
intt gsplit_f(void *pcxt, intt *m_split, intt pdim, intt sdim, intt rsize, intt idx_origin, intt idx_width, 
	intt nsplit, intt nstep, intt axis, bool bw)
{
	return gsplit_t<intt>(pcxt, m_split, pdim, sdim, rsize, idx_origin, idx_width, nsplit, nstep, axis, bw);
}

template<typename T>
__global__ void kconcat_f(void *pcxt, T *m_rcat, intt pdim, intt sdim, intt idx_origin, intt idx_width, 
	intt ncat, intt nstep, intt axis, bool bw, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	intt *prank;
	T *pcat_mdev, *pcat_mdevs;
	ConcatVar *ccv = (ConcatVar *)pcxt;

	prank = P_LINK_VAR2(intt, pcxt, ccv->szRankPrimary);
	pcat_mdevs = P_LINK_VAR2(T, pcxt, ccv->mptrDevSecondary);
	intt outer_sz = MRANK_SIZE(prank, axis), inner_sz = (axis == pdim - 1 ? 1 : MRANK_SIZE(prank, axis + 1)) * nstep;
	intt si, soff;

	soff = roff % outer_sz;//split축 랭크내의 분할 옵셋
	si = soff / inner_sz;//split축 분할 단위 이번 분할번째
	soff = (roff / outer_sz) * inner_sz + soff % inner_sz;//분할번째내의 옵셋
	pcat_mdev = *((T **)pcat_mdevs + si);

	if(bw) *(m_rcat + roff) += *(pcat_mdev + soff);
	else *(m_rcat + roff) = *(pcat_mdev + soff);
	//printf("[%d](%p) %d %d %d %d %f %f\n", roff, pcat_mdev, i, is, idx[axis], cat_off, *(pcat_mdev + cat_off), *(m_rcat + roff));
}
template<typename T>
intt gconcat_t(void *pcxt, T *m_rcat, intt pdim, intt sdim, intt rsize, intt idx_origin, intt idx_width, 
	intt ncat, intt nstep, intt axis, bool bw)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (rsize > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rsize);

	kconcat_f<T> << <grid, block >> > (pcxt, m_rcat, pdim, sdim, idx_origin, idx_width, ncat, nstep, axis, bw, n);
	cudaThreadSynchronize();

	return n - idx_origin * idx_width;
}
intt gconcat_f(void *pcxt, floatt *m_rcat, intt pdim, intt sdim, intt rsize, intt idx_origin, intt idx_width,
	intt ncat, intt nstep, intt axis, bool bw)
{
	return gconcat_t<floatt>(pcxt, m_rcat, pdim, sdim, rsize, idx_origin, idx_width, ncat, nstep, axis, bw);
}
intt gconcat_f(void *pcxt, intt *m_rcat, intt pdim, intt sdim, intt rsize, intt idx_origin, intt idx_width,
	intt ncat, intt nstep, intt axis, bool bw)
{
	return gconcat_t<intt>(pcxt, m_rcat, pdim, sdim, rsize, idx_origin, idx_width, ncat, nstep, axis, bw);
}
/*
template<typename T>
__global__ void kdot_f(void *pcxt, T *m_pdot, T *m_sdot, T *m_rdot, intt idx_origin, intt idx_width, T rplus, intt range, intt n)
{//쓰레드당 순차 범위 출력 차원 엑세스 버전
	DotVar *dotv = (DotVar *)pcxt;
	intt roff = idx_width * idx_origin + blockIdx.x * blockDim.x * range + threadIdx.x * range;
	//printf("%d %d %d\n", blockIdx.x, blockDim.x, threadIdx.x);
	if(roff >= n) return;
	if(n > roff + range) n = roff + range;
	
	intt *out_rank = dotv->outRank, nout = dotv->noutRank;
	intt njo_pre = dotv->njoPre, njo_suf = dotv->njoSuf;
	const bool bw_get_ori = dotv->bwGetOri, jdim_equal = dotv->jdimEqual;
	SparseRank *spr_out = dotv->sprPreOut, *spr_pre_jo = dotv->sprPreJo, *spr_suf_jo = dotv->sprSufJo;
	intt pj_idx[MX_DIM], sj_idx[MX_DIM], ret_idx[MX_DIM], i;
	register T sum;
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
	nout--;
	const intt njo_pre2 = njo_pre - 1, njo_suf2 = njo_suf - 1;
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
		if(bw_get_ori) {//역전파
			if(rplus != 1) *(m_rdot + roff) *= rplus;
			*(m_rdot + roff) += sum;
		} else {
			if(rplus == 0) *(m_rdot + roff) = 0;
			else *(m_rdot + roff) *= rplus;
			*(m_rdot + roff) += sum;
		}
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
}
template<typename T>
intt gdot_t(void *pcxt_dev, intt oksz, intt share_unit, T *pdot_mdev, T *sdot_mdev, T *rdot_mdev, intt rdot_size,
			intt idx_origin, intt idx_width, T rplus)
{
	//dim3 block(WIDTH_BLOCK);
	dim3 block(WIDTH_BLOCK2(SMALL_BLOCK));//현 dot는 BLOCK_SIZE 이던 idx_width이던 512를 넘어가면 즉 동시 수행 쓰레드 갯수가
	intt range = idx_width / block.x;//512를 넘어가면 로컬 메모리(변수) 사이즈가 한계 초과되어 실행되지 않는다. 
	dim3 grid(idx_width % (block.x * range) ? 2 : 1);
	intt n = (rdot_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rdot_size);

	kdot_f<T> << <grid, block >> > (pcxt_dev, pdot_mdev, sdot_mdev, rdot_mdev, idx_origin, idx_width, rplus, range, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}*/
/*
template<typename T>
__global__ void kdot_f(void *pcxt, T *m_pdot, T *m_sdot, T *m_rdot, intt idx_origin, intt idx_width, T rplus, intt n)
{//루프없이 one thread one out버전, 복수 grid실행
	DotVar *dotv = (DotVar *)pcxt;
	intt roff = idx_width * idx_origin + blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d %d %d\n", blockIdx.x, blockDim.x, threadIdx.x);
	if(roff >= n) return;

	SparseRank *spr_pre_jo = dotv->sprPreJo, *spr_suf_jo = dotv->sprSufJo;
	intt pj_idx[MX_DIM], sj_idx[MX_DIM], i;
	register T sum;
	//현 roff는 ret매트릭스가 기준인 옵셋이고 이것을 이번 매트릭스 곱의 양측 out axis rank기준으로 변환한다.
	_offset2idx(dotv->noutRank, dotv->outRank, roff, pj_idx);
	for(i = 0;i < dotv->noutRank; i++) {
		if(dotv->sprPreOut[i].rkPref) m_pdot += (pj_idx[i] * dotv->sprPreOut[i].rksz);
		else m_sdot += (pj_idx[i] * dotv->sprPreOut[i].rksz);
	}
	if(dotv->jdimEqual) {//양측 조인 랭크가 같으면 한번에 초기화
		for(i = 0;i < dotv->njoPre; i++) pj_idx[i] = sj_idx[i] = 0;
	} else {
		for(i = 0;i < dotv->njoPre; i++) pj_idx[i] = 0;
		for(i = 0;i < dotv->njoSuf; i++) sj_idx[i] = 0;
	}
	const intt njo_pre2 = dotv->njoPre - 1, njo_suf2 = dotv->njoSuf - 1;
	const intt pjdim = spr_pre_jo[njo_pre2].rkdim, pjsz = spr_pre_jo[njo_pre2].rksz;
	const intt sjdim = spr_suf_jo[njo_suf2].rkdim, sjsz = spr_suf_jo[njo_suf2].rksz;
	intt i_pj = 0, i_sj = 0;

	for(sum = 0;;) {//끝에서부터 계수, 조인 인덱스만 순차적으로 증가하며 양측 원소 곱셈, 조인트 차원 인덱스 설정
		//printf("%f %f\n", *m_pdot, *m_sdot);
		sum += *m_pdot * *m_sdot;
		if(dotv->jdimEqual) {//양측 조인 인덱스가 같으면 대표로 pre join인덱스로 제어
			for(;;) {//pre 조인차원 인덱스 증가
				if(pjdim == ++i_pj) {//단위(차원) 올림과 동시에 마지막에서 조인 계수 초기화가 자동으로 된다.
					i_pj = 0;
					i = njo_pre2;
J0:;				m_pdot -= spr_pre_jo[i].rktsz;
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
J1:;				m_pdot -= spr_pre_jo[i].rktsz;
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
	if(dotv->bwGetOri) {//역전파
		if(rplus != 1) *(m_rdot + roff) *= rplus;
		*(m_rdot + roff) += sum;
	} else {
		if(rplus == 0) *(m_rdot + roff) = 0;
		else *(m_rdot + roff) *= rplus;
		*(m_rdot + roff) += sum;
	}
}
template<typename T>
intt gdot_t(void *pcxt_dev, intt oksz, intt share_unit, T *pdot_mdev, T *sdot_mdev, T *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, T rplus)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (rdot_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rdot_size);

	kdot_f<T> << <grid, block >> > (pcxt_dev, pdot_mdev, sdot_mdev, rdot_mdev, idx_origin, idx_width, rplus, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}*/
/*
template<typename T>
__global__ void kdot_f(void *pcxt, T *_m_pdot, T *_m_sdot, T *m_rdot, intt idx_origin, intt idx_width, T rplus, intt n)
{//조인 축 메모리 결합 엑세스 버전: 효과없음, 나중에 혹시 사용할려면 jdimEqual아닐때 결과 틀리므로 jdimEqual일때만 
	//결합으로 수행하게 jdimEqual아닐때는 결합 수횅안하게 결합 코드 제거, 백워드 과정은 테스트 안했음.
	//__shared__ T cache_bank[SM_SIZE];
	DotVar *dotv = (DotVar *)pcxt;
	intt roff = idx_width * idx_origin + blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d %d %d\n", blockIdx.x, blockDim.x, threadIdx.x);
	if(roff >= n) return;

	SparseRank *spr_pre_jo = dotv->sprPreJo, *spr_suf_jo = dotv->sprSufJo;
	const intt njo_pre2 = dotv->njoPre - 1, njo_suf2 = dotv->njoSuf - 1;
	intt pjdim = spr_pre_jo[njo_pre2].rkdim, pjsz = spr_pre_jo[njo_pre2].rksz;
	intt sjdim = spr_suf_jo[njo_suf2].rkdim, sjsz = spr_suf_jo[njo_suf2].rksz;
	intt pj_idx[MX_DIM], sj_idx[MX_DIM], tmp_idx[MX_DIM], i, i_pj, i_sj;
	bool pover = 0, sover = 0;
	register T sum;

	if(dotv->jdimEqual) {//양측 조인 랭크가 같으면 한번에 초기화
		for(i = 0;i < dotv->njoPre; i++) pj_idx[i] = sj_idx[i] = 0;
	} else {
		for(i = 0;i < dotv->njoPre; i++) pj_idx[i] = 0;
		for(i = 0;i < dotv->njoSuf; i++) sj_idx[i] = 0;
	}
	T *m_pdot, *m_sdot;
	for(;roff < n; roff += blockDim.x) {
		//__syncthreads();
		_offset2idx(dotv->noutRank, dotv->outRank, roff, tmp_idx);
		for(i = 0, m_pdot = _m_pdot, m_sdot = _m_sdot;i < dotv->noutRank; i++) {
			if(dotv->sprPreOut[i].rkPref) m_pdot += (tmp_idx[i] * dotv->sprPreOut[i].rksz);
			else m_sdot += (tmp_idx[i] * dotv->sprPreOut[i].rksz);
		}
		i_pj = (blockDim.x < pjdim ? threadIdx.x : threadIdx.x % pjdim);
		if(i_pj == 0) pover = 1;
		if(dotv->jdimEqual) i_sj = i_pj;
		else {
			i_sj = (blockDim.x < sjdim ? threadIdx.x : threadIdx.x % sjdim);
			if(i_sj == 0) sover = 1;
		}
		m_pdot += i_pj * pjsz;
		m_sdot += i_sj * sjsz;
		for(sum = 0;;) {//끝에서부터 계수, 조인 인덱스만 순차적으로 증가하며 양측 원소 곱셈, 조인트 차원 인덱스 설정
			//if(threadIdx.x == 2) printf("%d %d %d %f %f\n", threadIdx.x, i_pj, pjdim, *m_pdot, *m_sdot);
			sum += *m_pdot * *m_sdot;
			if(dotv->jdimEqual) {//양측 조인 인덱스가 같으면 대표로 pre join인덱스로 제어
				for(;;) {//pre 조인차원 인덱스 증가
					if(pjdim == ++i_pj) {//단위(차원) 올림과 동시에 마지막에서 조인 계수 초기화가 자동으로 된다.
						if(pover) {
							m_pdot += (pjsz * (spr_pre_jo[njo_pre2].rkdim - pjdim));
							m_sdot += (sjsz * (spr_pre_jo[njo_pre2].rkdim - pjdim));
							pjdim = spr_pre_jo[njo_pre2].rkdim;
							pover = 0;
							i = njo_pre2;
J0:;						m_pdot -= spr_pre_jo[i].rktsz;
							m_sdot -= spr_suf_jo[i].rktsz;
							if(--i < 0) goto LB1;
							else {//중간 차원 증가
								if(spr_pre_jo[i].rkdim == ++pj_idx[i]) {
									pj_idx[i] = 0;
									goto J0;
								} else {
									i_pj = (blockDim.x < pjdim ? threadIdx.x : threadIdx.x % pjdim);
									if(i_pj == 0) pover = 1;
									m_pdot += (spr_pre_jo[i].rksz + i_pj * pjsz);//다음 상위 단위(차원)값 증가
									m_sdot += (spr_suf_jo[i].rksz + i_pj * sjsz);
									break;
								}
							}
						} else {
							i_pj = 0;
							pover = 1;
							pjdim = (blockDim.x < pjdim ? threadIdx.x : threadIdx.x % pjdim);
							m_pdot -= spr_pre_jo[njo_pre2].rktsz;//m_pdot -= (spr_pre_jo[njo_pre2].rktsz - (pjdim * pjsz));
							m_sdot -= spr_suf_jo[njo_pre2].rktsz;//m_sdot -= (spr_suf_jo[njo_pre2].rktsz - (pjdim * sjsz));
							break;
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
						if(pover) {
							m_pdot += (pjsz * (spr_pre_jo[njo_pre2].rkdim - pjdim));
							pjdim = spr_pre_jo[njo_pre2].rkdim;
							pover = 0;
							i = njo_pre2;
J1:;						m_pdot -= spr_pre_jo[i].rktsz;
							if(--i < 0) break;
							else {
								if(spr_pre_jo[i].rkdim == ++pj_idx[i]) {
									pj_idx[i] = 0;
									goto J1;
								} else {
									i_pj = (blockDim.x < pjdim ? threadIdx.x : threadIdx.x % pjdim);
									if(i_pj == 0) pover = 1;
									m_pdot += (spr_pre_jo[i].rksz + i_pj * pjsz);//다음 상위 단위(차원)값 증가
									break;
								}
							}
						} else {
							i_pj = 0;
							pover = 1;
							pjdim = (blockDim.x < pjdim ? threadIdx.x : threadIdx.x % pjdim);
							m_pdot -= spr_pre_jo[njo_pre2].rktsz;//m_pdot -= (spr_pre_jo[njo_pre2].rktsz - (pjdim * pjsz));
							break;
						}
					} else {
						m_pdot += pjsz;//현 단위(차원)값 증가
						break;
					}
				}
				for(;;) {//suf 조인차원 인덱스 증가
					if(sjdim == ++i_sj) {
						if(sover) {
							m_sdot += (sjsz * (spr_suf_jo[njo_suf2].rkdim - sjdim));
							sjdim = spr_suf_jo[njo_suf2].rkdim;
							sover = 0;
							i = njo_suf2;
J2:;						m_sdot -= spr_suf_jo[i].rktsz;
							if(--i < 0) goto LB1;
							else {
								if(spr_suf_jo[i].rkdim == ++sj_idx[i]) {
									sj_idx[i] = 0;
									goto J2;
								} else {
									i_sj = (blockDim.x < sjdim ? threadIdx.x : threadIdx.x % sjdim);
									if(i_sj == 0) sover = 1;
									m_sdot += (spr_suf_jo[i].rksz + i_sj * sjsz);//다음 상위 단위(차원)값 증가
									break;
								}
							}
						} else {
							i_sj = 0;
							sover = 1;
							sjdim = (blockDim.x < sjdim ? threadIdx.x : threadIdx.x % sjdim);
							m_sdot -= spr_suf_jo[njo_suf2].rktsz;//m_sdot -= (spr_suf_jo[njo_suf2].rktsz - (sjdim * sjsz));
							break;
						}
					} else {
						m_sdot += sjsz;
						break;
					}
				}
			}
		}
LB1:;
		if(dotv->bwGetOri) {//역전파
			if(rplus != 1) *(m_rdot + roff) *= rplus;
			*(m_rdot + roff) += sum;
		} else {
			if(rplus == 0) *(m_rdot + roff) = 0;
			else *(m_rdot + roff) *= rplus;
			*(m_rdot + roff) += sum;
		}
	}
}
template<typename T>
intt gdot_t(void *pcxt_dev, intt oksz, intt share_unit, T *pdot_mdev, T *sdot_mdev, T *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, T rplus)
{
	dim3 block(WIDTH_BLOCK2(SMALL_BLOCK));
	//dim3 block(WIDTH_BLOCK);
	intt n = (rdot_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rdot_size);
	kdot_f<T> << <1, block >> > (pcxt_dev, pdot_mdev, sdot_mdev, rdot_mdev, idx_origin, idx_width, rplus, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}*/
/*
template<typename T>
__global__ void kdot_f(void *pcxt, T *_m_pdot, T *_m_sdot, T *m_rdot, intt idx_origin, intt idx_width, T rplus, intt n)
{//쓰레드당 출력 축 한개 주소 엑세스, 쓰레드당 블럭 단위 주조 점프 루프, 좌표 변환으로 획득 버전.
	//__shared__ T cache_bank[SM_SIZE];
	DotVar *dotv = (DotVar *)pcxt;
	intt roff = idx_width * idx_origin + blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d %d %d\n", blockIdx.x, blockDim.x, threadIdx.x);
	if(roff >= n) return;

	SparseRank *spr_pre_jo = dotv->sprPreJo, *spr_suf_jo = dotv->sprSufJo;
	const intt njo_pre2 = dotv->njoPre - 1, njo_suf2 = dotv->njoSuf - 1;
	const intt pjdim = spr_pre_jo[njo_pre2].rkdim, pjsz = spr_pre_jo[njo_pre2].rksz;
	const intt sjdim = spr_suf_jo[njo_suf2].rkdim, sjsz = spr_suf_jo[njo_suf2].rksz;
	intt pj_idx[MX_DIM], sj_idx[MX_DIM], tmp_idx[MX_DIM], i, i_pj, i_sj;
	register T sum;

	if(dotv->jdimEqual) {//양측 조인 랭크가 같으면 한번에 초기화
		for(i = 0;i < dotv->njoPre; i++) pj_idx[i] = sj_idx[i] = 0;
	} else {
		for(i = 0;i < dotv->njoPre; i++) pj_idx[i] = 0;
		for(i = 0;i < dotv->njoSuf; i++) sj_idx[i] = 0;
	}
	T *m_pdot, *m_sdot;
	for(;roff < n; roff += blockDim.x) {
		//__syncthreads();
		_offset2idx(dotv->noutRank, dotv->outRank, roff, tmp_idx);
		for(i = 0, m_pdot = _m_pdot, m_sdot = _m_sdot;i < dotv->noutRank; i++) {
			if(dotv->sprPreOut[i].rkPref) m_pdot += (tmp_idx[i] * dotv->sprPreOut[i].rksz);
			else m_sdot += (tmp_idx[i] * dotv->sprPreOut[i].rksz);
		}
		for(sum = 0, i_pj = i_sj = 0;;) {//끝에서부터 계수, 조인 인덱스만 순차적으로 증가하며 양측 원소 곱셈, 조인트 차원 인덱스 설정
			//printf("%f %f\n", *m_pdot, *m_sdot);
			sum += *m_pdot * *m_sdot;
			if(dotv->jdimEqual) {//양측 조인 인덱스가 같으면 대표로 pre join인덱스로 제어
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
		if(dotv->bwGetOri) {//역전파
			if(rplus != 1) *(m_rdot + roff) *= rplus;
			*(m_rdot + roff) += sum;
		} else {
			if(rplus == 0) *(m_rdot + roff) = 0;
			else *(m_rdot + roff) *= rplus;
			*(m_rdot + roff) += sum;
		}
	}
}
template<typename T>
intt gdot_t(void *pcxt_dev, intt oksz, intt share_unit, T *pdot_mdev, T *sdot_mdev, T *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, T rplus)
{
	dim3 block(WIDTH_BLOCK2(SMALL_BLOCK));
	//dim3 block(WIDTH_BLOCK);
	intt n = (rdot_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rdot_size);
	kdot_f<T> << <1, block >> > (pcxt_dev, pdot_mdev, sdot_mdev, rdot_mdev, idx_origin, idx_width, rplus, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}*/
/*
template<typename T>
__global__ void kdot_f(void *pcxt, T *_m_pdot, T *_m_sdot, T *m_rdot, intt idx_origin, intt idx_width, T rplus, intt n)
{//쓰레드당 출력 축 한개 주소 엑세스, 쓰레드당 블럭 단위 주조 점프 루프, 좌표 변환으로 획득, 공유메모리사용, 한개 블럭 실행 버전.
	__shared__ T _cache_bank[SM_SIZE];
	T *cache_bank;
	DotVar *dotv = (DotVar *)pcxt;
	intt roff = idx_width * idx_origin + blockIdx.x * blockDim.x + threadIdx.x;
	intt roff_ori = roff;
	//printf("qq %d %d %d %d %d %d\n", idx_width, idx_origin, blockIdx.x, blockDim.x, threadIdx.x, roff);
	if(roff >= n) return;

	SharePoint *sp = (SharePoint *)&_cache_bank[0];
	if(threadIdx.x == 0) {
		memcpy(sp->spr_out, dotv->sprPreOut, sizeof(SparseRank) * dotv->noutRank);
		sp->cache_axis = dotv->axisCache;
		sp->okfit = dotv->fitOutKernel;
		sp->jkfit = dotv->fitJoKernel;
		sp->oksz = blockDim.x;//출력 매트릭스 크기가 출력 커널 사이즈보다 작으면 출력 매트릭스 크기로 설정됨.
		sp->jksz = dotv->szJoKernel;
		sp->nrecycc = dotv->nrecycCache;
		sp->ncycj = dotv->ncycJo;
		sp->njoint = dotv->nJointAxis;
		sp->nout = dotv->noutRank;
		memcpy(sp->out_rank, dotv->outRank, sizeof(intt) * dotv->noutRank);
		if(sp->spr_out[sp->cache_axis].rkPref) {//연속하는 최 하위차원 바로위의 (출력)차원은 각 원소이 조인 차원 원소들을 
			memcpy(sp->spr_jcache, dotv->sprPreJo, sizeof(SparseRank) * dotv->njoPre);	//공유메모리에 캐쉬하여 그 하위(출력)차원 한개 원소마다 조인 순열을 재사용한다.
			memcpy(sp->jrank_cache, dotv->joRankPre, sizeof(intt) * dotv->njoPre);
			sp->njo_cache = dotv->njoPre;
			memcpy(sp->spr_jleaf, dotv->sprSufJo, sizeof(SparseRank) * dotv->njoSuf);//출력에서 연속하는 최 하위 단말 차원(연속하지 않으면 pre와 suf중 한개 차원)
			memcpy(sp->jrank_leaf, dotv->joRankSuf, sizeof(intt) * dotv->njoSuf);
			sp->njo_leaf = dotv->njoSuf;
		} else {
			memcpy(sp->spr_jcache, dotv->sprSufJo, sizeof(SparseRank) * dotv->njoSuf);
			memcpy(sp->jrank_cache, dotv->joRankSuf, sizeof(intt) * dotv->njoSuf);
			sp->njo_cache = dotv->njoSuf;
			memcpy(sp->spr_jleaf, dotv->sprPreJo, sizeof(SparseRank) * dotv->njoPre);
			memcpy(sp->jrank_leaf, dotv->joRankPre, sizeof(intt) * dotv->njoPre);
			sp->njo_leaf = dotv->njoPre;
		}
	}
	__syncthreads();
	//printf("zz %d %d\n", sp->jksz, dotv->szJoKernel);
	cache_bank = (T *)((bytet *)_cache_bank + sizeof(SharePoint));
	cache_bank = (T *)ALIGN_UNIT((divadx)cache_bank);
	const intt njo_leaf2 = sp->njo_leaf - 1;
	intt ljdim = sp->spr_jleaf[njo_leaf2].rkdim;
	const intt ljdim_ori = ljdim, ljsz = sp->spr_jleaf[njo_leaf2].rksz;
	intt leaf_idx[MX_DIM], tmp_idx[MX_DIM], i, i_lj, j;
	register T sum;

	T *m_pdot, *m_sdot, *m_cache, *m_leaf;
	intt inc = blockDim.x, itime_ok = sp->nrecycc, itime_jk = 0;//초기 횟수값 설정.
	//intt iloop = 0;
	for(;; roff += inc) {
LP:;	//printf("aa %d %d %d\n", roff, itime_ok, sp->nrecycc);
		cache_load(n, roff_ori, roff, sp->okfit, sp->jkfit, sp->oksz, sp->jksz, sp->nrecycc, sp->ncycj, 
			sp->njoint, itime_ok, itime_jk, threadIdx.x, blockDim.x, inc, sp->cache_axis, sp->nout, sp->out_rank, tmp_idx,
			sp->spr_out, sp->spr_jcache, sp->spr_jleaf, sp->njo_cache, sp->jrank_cache, sp->njo_leaf, 
			sp->jrank_leaf,	_m_pdot, m_pdot, _m_sdot, m_sdot, m_cache, m_leaf, leaf_idx, cache_bank, LP);
		i_lj = leaf_idx[njo_leaf2];//이번 조인 커널 최말단 시작 인덱스
		//printf("sss %d: %d %d %d\n", threadIdx.x, ljdim, i_lj);
		if(sp->jksz < ljdim - i_lj) i_lj = ljdim - sp->jksz;//조인되는 갯수가 조인커널 사이즈(jksz)보다 적은 경우는 ㄱ)에서 체크되므로 
		//고려하지않고 조인사이즈가 커서 분할되어 jksz이 최하단말 사이즈모다 작은경우 ㄴ)에서 체크되게 설정.
		for(sum = 0, j = 0;;) {//끝에서부터 계수, 조인 인덱스만 순차적으로 증가하며 양측 원소 곱셈, 조인트 차원 인덱스 설정
			//printf("%d: %d %d %f %f\n", threadIdx.x, sp->jksz, ljsz, *m_cache, *m_leaf);
			sum += *m_cache * *m_leaf;
			//printf("ss %d: %d %d\n", threadIdx.x, ljdim, i_lj);
			if(ljdim == ++i_lj) {//ㄴ.최하 단말 차원 인덱스 증가
				if(sp->jksz + leaf_idx[njo_leaf2] < (++j + 1) * ljdim) {//조인커널사이즈가 초기시작 갯수에서 누적하여 다음번 최하말단
					i_lj = (j +1) * ljdim - (sp->jksz + leaf_idx[njo_leaf2]);//체크사이즈보다 적으면 최하말단 사이즈롤 
					if(i_lj >= ljdim) break;									//적으만큼 설정하여 다음에 ㄴ)에서 체크되게 한다.
				} else i_lj = 0;
				i = njo_leaf2;
J2:;			m_leaf -= sp->spr_jleaf[i].rktsz;
				if(--i < 0) {
					//printf("vv %d: %d %d\n", threadIdx.x, ljdim, i_lj);
					goto LB1;//ㄱ.
				} else {
					if(sp->spr_jleaf[i].rkdim == ++leaf_idx[i]) {
						leaf_idx[i] = 0;
						goto J2;
					} else m_leaf += sp->spr_jleaf[i].rksz;//다음 상위 단위(차원)값 증가
				}
			} else m_leaf += ljsz;
			m_cache++;
		}
LB1:;
		if(dotv->bwGetOri) {//역전파
			if(itime_jk == 0 && rplus != 1) *(m_rdot + roff) *= rplus;
			*(m_rdot + roff) += sum;
		} else {
			if(itime_jk == 0) {
				if(rplus == 0) *(m_rdot + roff) = 0;
				else *(m_rdot + roff) *= rplus;
			}
			*(m_rdot + roff) += sum;
			//printf("%f\n", sum);
		}
		//iloop++;
	}
	//printf("## %d\n", iloop);
}
template<typename T>
intt gdot_t(void *pcxt_dev, intt oksz, intt share_unit, T *pdot_mdev, T *sdot_mdev, T *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, T rplus)
{
	//dim3 block(WIDTH_BLOCK2(SMALL_BLOCK));
	//dim3 block(WIDTH_BLOCK);
	dim3 block(WIDTH_BLOCK2(oksz));//설정된 출력 커널 사이즈보다 블럭 사이즈가 더 크게 설정되지 않는다.
	intt n = (rdot_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rdot_size);
	kdot_f<T> << <1, block >> > (pcxt_dev, pdot_mdev, sdot_mdev, rdot_mdev, idx_origin, idx_width, rplus, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}*/
template<typename T>
__global__ void kdot_f(void *pcxt, intt share_unit, T *_m_pdot, T *_m_sdot, T *m_rdot, intt idx_origin, intt idx_width, T rplus, intt n)
{//쓰레드당 출력 축 한개 주소 엑세스, 쓰레드당 블럭 단위 주조 점프 루프, 좌표 변환으로 획득, 공유메모리사용, 복수 블럭 실행 버전.
	__shared__ T _cache_bank[SM_SIZE];//나중에 공유메모리 용량이 커지면 m_leaf매트릭스 전채와 leaf_idx, tmp_idx를 공유메모리에 적재한다.
	T *cache_bank;
	DotVar *dotv = (DotVar *)pcxt;
	intt roff = idx_width * idx_origin + blockIdx.x * share_unit + threadIdx.x;
	intt roff_ori = roff;
	//printf("qq %d %d %d %d %d %d\n", idx_width, idx_origin, blockIdx.x, blockDim.x, threadIdx.x, roff);
	if(roff >= n) return;
	
	share_unit = roff - threadIdx.x + share_unit;//share_unit은 바로 아래서 사용할려고 변경, 이후에는 사용되지 않으므로
	if(n > share_unit) n = share_unit;

	SharePoint *sp = (SharePoint *)&_cache_bank[0];
	if(threadIdx.x == 0) {
		memcpy(sp->spr_out, dotv->sprPreOut, sizeof(SparseRank) * dotv->noutRank);
		sp->cache_axis = dotv->axisCache;
		sp->okfit = dotv->fitOutKernel;
		sp->jkfit = dotv->fitJoKernel;
		sp->oksz = blockDim.x;//출력 매트릭스 크기가 출력 커널 사이즈보다 작으면 출력 매트릭스 크기로 설정됨.
		sp->jksz = dotv->szJoKernel;
		sp->nrecycc = dotv->nrecycCache;
		sp->ncycj = dotv->ncycJo;
		sp->njoint = dotv->nJointAxis;
		sp->nout = dotv->noutRank;
		memcpy(sp->out_rank, dotv->outRank, sizeof(intt) * dotv->noutRank);
		if(sp->spr_out[sp->cache_axis].rkPref) {//연속하는 최 하위차원 바로위의 (출력)차원은 각 원소이 조인 차원 원소들을 
			memcpy(sp->spr_jcache, dotv->sprPreJo, sizeof(SparseRank) * dotv->njoPre);	//공유메모리에 캐쉬하여 그 하위(출력)차원 한개 원소마다 조인 순열을 재사용한다.
			memcpy(sp->jrank_cache, dotv->joRankPre, sizeof(intt) * dotv->njoPre);
			sp->njo_cache = dotv->njoPre;
			memcpy(sp->spr_jleaf, dotv->sprSufJo, sizeof(SparseRank) * dotv->njoSuf);//출력에서 연속하는 최 하위 단말 차원(연속하지 않으면 pre와 suf중 한개 차원)
			memcpy(sp->jrank_leaf, dotv->joRankSuf, sizeof(intt) * dotv->njoSuf);
			sp->njo_leaf = dotv->njoSuf;
		} else {
			memcpy(sp->spr_jcache, dotv->sprSufJo, sizeof(SparseRank) * dotv->njoSuf);
			memcpy(sp->jrank_cache, dotv->joRankSuf, sizeof(intt) * dotv->njoSuf);
			sp->njo_cache = dotv->njoSuf;
			memcpy(sp->spr_jleaf, dotv->sprPreJo, sizeof(SparseRank) * dotv->njoPre);
			memcpy(sp->jrank_leaf, dotv->joRankPre, sizeof(intt) * dotv->njoPre);
			sp->njo_leaf = dotv->njoPre;
		}
	}
	__syncthreads();
	//printf("zz %d %d\n", sp->jksz, dotv->szJoKernel);
	cache_bank = (T *)((bytet *)_cache_bank + sizeof(SharePoint));
	cache_bank = (T *)ALIGN_UNIT((divadx)cache_bank);
	const intt njo_leaf2 = sp->njo_leaf - 1;
	intt ljdim = sp->spr_jleaf[njo_leaf2].rkdim;
	const intt ljsz = sp->spr_jleaf[njo_leaf2].rksz;
	intt leaf_idx[MX_DIM], tmp_idx[MX_DIM], i, i_lj, j;
	bool bw_get_ori = dotv->bwGetOri;
	register T sum;

	T *m_pdot, *m_sdot, *m_cache, *m_leaf;
	intt inc = blockDim.x, itime_ok = sp->nrecycc, itime_jk = 0;//초기 횟수값 설정.
	//intt iloop = 0;
	for(;; roff += inc) {
LP:;	//printf("aa %d %d %d\n", roff, itime_ok, sp->nrecycc);
		cache_load(n, roff_ori, roff, sp->okfit, sp->jkfit, sp->oksz, sp->jksz, sp->nrecycc, sp->ncycj,
			sp->njoint, itime_ok, itime_jk, threadIdx.x, blockDim.x, inc, sp->cache_axis, sp->nout, sp->out_rank, tmp_idx,
			sp->spr_out, sp->spr_jcache, sp->spr_jleaf, sp->njo_cache, sp->jrank_cache, sp->njo_leaf,
			sp->jrank_leaf, _m_pdot, m_pdot, _m_sdot, m_sdot, m_cache, m_leaf, leaf_idx, cache_bank, LP);
		i_lj = leaf_idx[njo_leaf2];//이번 조인 커널 최말단 시작 인덱스
		//printf("sss %d: %d %d %d\n", threadIdx.x, ljdim, i_lj);
		if(sp->jksz < ljdim - i_lj) i_lj = ljdim - sp->jksz;//조인되는 갯수가 조인커널 사이즈(jksz)보다 적은 경우는 ㄱ)에서 체크되므로 
		//고려하지않고 조인사이즈가 커서 분할되어 jksz이 최하단말 사이즈모다 작은경우 ㄴ)에서 체크되게 설정.
		for(sum = 0, j = 0;;) {//끝에서부터 계수, 조인 인덱스만 순차적으로 증가하며 양측 원소 곱셈, 조인트 차원 인덱스 설정
			//printf("%d: %d %d %f %f\n", threadIdx.x, sp->jksz, ljsz, *m_cache, *m_leaf);
			sum += *m_cache * *m_leaf;
			//printf("ss %d: %d %d\n", threadIdx.x, ljdim, i_lj);
			if(ljdim == ++i_lj) {//ㄴ.최하 단말 차원 인덱스 증가
				if(sp->jksz + leaf_idx[njo_leaf2] < (++j + 1) * ljdim) {//조인커널사이즈가 초기시작 갯수에서 누적하여 다음번 최하말단
					i_lj = (j + 1) * ljdim - (sp->jksz + leaf_idx[njo_leaf2]);//체크사이즈보다 적으면 최하말단 사이즈롤 
					if(i_lj >= ljdim) break;									//적으만큼 설정하여 다음에 ㄴ)에서 체크되게 한다.
				} else i_lj = 0;
				i = njo_leaf2;
J2:;			m_leaf -= sp->spr_jleaf[i].rktsz;
				if(--i < 0) {
					//printf("vv %d: %d %d\n", threadIdx.x, ljdim, i_lj);
					goto LB1;//ㄱ.
				} else {
					if(sp->spr_jleaf[i].rkdim == ++leaf_idx[i]) {
						leaf_idx[i] = 0;
						goto J2;
					} else m_leaf += sp->spr_jleaf[i].rksz;//다음 상위 단위(차원)값 증가
				}
			} else m_leaf += ljsz;
			m_cache++;
		}
LB1:;
		if(itime_jk) *(m_rdot + roff) += sum;
		else {
			if(rplus) *(m_rdot + roff) += sum;
			else *(m_rdot + roff) = sum;
		}
		//iloop++;
	}
	//printf("## %d\n", iloop);
}
template<typename T>
intt gdot_t(void *pcxt_dev, intt oksz, intt share_unit, T *pdot_mdev, T *sdot_mdev, T *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, T rplus)
{
	//dim3 block(WIDTH_BLOCK2(SMALL_BLOCK));
	//dim3 block(WIDTH_BLOCK);
	dim3 block(WIDTH_BLOCK2(oksz));//설정된 출력 커널 사이즈보다 블럭 사이즈가 더 크게 설정되지 않는다.
	dim3 grid((idx_width + share_unit - 1) / share_unit);//그리드 사이즈를 줄일려면 아래같이 share_unit 배수로 
	//grid((idx_width + share_unit *2 - 1) / share_unit *2);//계산(x배수로 설정하면 블럭단위에서 share_unit x개가 처리된다.)하면 되나 policyTrack에서도 같이 처리되야 하므로 원 share_unit이 x배가 되게한다.
	intt n = (rdot_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rdot_size);
	kdot_f<T> << <grid, block >> > (pcxt_dev, share_unit, pdot_mdev, sdot_mdev, rdot_mdev, idx_origin, idx_width, rplus, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gdot_f(void *pcxt_dev, intt oksz, intt share_unit, floatt *pdot_mdev, floatt *sdot_mdev, floatt *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, floatt rplus)
{
	return gdot_t<floatt>(pcxt_dev, oksz, share_unit, pdot_mdev, sdot_mdev, rdot_mdev, rdot_size,
		idx_origin, idx_width, rplus);
}
intt gdot_f(void *pcxt_dev, intt oksz, intt share_unit, intt *pdot_mdev, intt *sdot_mdev, intt *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, intt rplus)
{
	return gdot_t<intt>(pcxt_dev, oksz, share_unit, pdot_mdev, sdot_mdev, rdot_mdev, rdot_size,
		idx_origin, idx_width, rplus);
}
template<typename T>
__global__ void karith_f(void *pcxt, T *m_pari, T *m_sari, T *m_rari, intt idx_origin, intt idx_width, T sval, sytet aop, T rplus, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	ArithVar *arv = (ArithVar *)pcxt;
	intt *mrank = arv->arRankMast, *prank = arv->arRankPre, *srank = arv->arRankSuf, *rrank = arv->arRankRet;
	intt npre = arv->narPre, nsuf = arv->narSuf, nmast = arv->narMast;
	intt off, cidx[MX_DIM], tmp_idx[MX_DIM], coff;
	T rval, *ppre, *psuf;

	if(arv->bwGetOri) {
		dlead_offset2idx(arv->narBro, npre, arv->narRet, rrank, roff, cidx);
		for(;;) {//rrank의 한개 원소에 대하여 rrank의 차원이 1인 랭크들을 순차적으로 순회하여 브로드캐스팅된 것들의 pre와 suf 연산
			coff = dbro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx);
			if(coff < 0) break;
			if(m_pari) {//역전파에서 pref는 순전파에서 리턴되는 브로드캐스트된 매트릭스로 위에서 mrank로 옵셋계산되므로
				ppre = m_pari + coff;//바로 사용.
				//printf("%d ", coff);
			} else ppre = &sval;
			if(m_sari) {
				off = dmoff2soff(nmast, mrank, nsuf, srank, coff, tmp_idx);
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
				else if(*(m_rari + roff) != rval) printf("xxx\n");
			} else {
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
		}
	} else {
		if(m_pari) {
			off = dmoff2soff(nmast, mrank, npre, prank, roff, tmp_idx);
			ppre = m_pari + off;
			//printf("%d ", off);
		} else ppre = &sval;
		if(m_sari) {
			off = dmoff2soff(nmast, mrank, nsuf, srank, roff, tmp_idx);
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
template<typename T>
__global__ void karith_f1(void *pcxt, T *m_pari, T *m_sari, T *m_rari, intt idx_origin, intt idx_width, sytet aop, T rplus, intt n)
{//브로드 케스트가 없는 양측 매트릭스 일대일 대응 
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	ArithVar *arv = (ArithVar *)pcxt;
	T rval;//pre나 suf가 ret와 겹치면 pre나 suf가 계산되기위해 참조되기 전에 리셋되므로 먼저 게산하고 보관하기위해
	if(arv->bwGetOri) {
		switch(aop) {
		case AOP_MUL:
			rval = *(m_pari + roff) * *(m_sari + roff);
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case AOP_PLUS:
			rval = *(m_pari + roff);
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case AOP_DIV:
			break;
		case AOP_MINUS:
			break;
		case ABP_MINUS_PREF:
			rval = *(m_pari + roff);
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case ABP_MINUS_SUFF:
			rval = *(m_pari + roff) * -1;
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case ABP_DIV_PREF:
			rval = *(m_pari + roff) * (1 / *(m_sari + roff));
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case ABP_DIV_SUFF:
			rval = *(m_pari + roff) * (1 / (*(m_sari + roff) * *(m_sari + roff)) * -1);
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case ABP_BWTEST:
			rval = *(m_pari + roff) / *(m_sari + roff);
			if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
			else if(*(m_rari + roff) != rval)  printf("xxx\n");
			break;
		}
	} else {
		switch(aop) {
		case AOP_MUL:
			rval = *(m_pari + roff) * *(m_sari + roff);
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_PLUS:
			rval = *(m_pari + roff) + *(m_sari + roff);
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_DIV:
			rval = *(m_pari + roff) / *(m_sari + roff);
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_MINUS:
			rval = *(m_pari + roff) - *(m_sari + roff);
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		}
	}
}
template<typename T>
__global__ void karith_f2_bwprem(void *pcxt, T *m_pari, T *m_sari, T *m_rari, intt idx_origin, intt idx_width, sytet aop, T rplus, intt n)
{//어느 한쪽이 작은 브로드 케스트(디멘젼이 1인 차원이 없는) 백워드, m_par사이즈가 m_rari사이즈 보다 커서 m_par를 그리드블럭으로
	intt poff = blockIdx.x*blockDim.x + threadIdx.x;
	ArithVar *arv = (ArithVar *)pcxt;
	intt osz = arv->zarOut, ssz = arv->zarSuf, psz = arv->zarPre;
	intt roff = poff % osz;

	if(roff < idx_width * idx_origin || roff >= n) return;//리턴 매트릭스가 분할되었을 경루 pre로부터 산출한 roff의 범위 체크

	T rval;
	switch(aop) {
	case AOP_MUL:
		rval = *(m_pari + poff) * *(m_sari + poff % ssz);
		if(rplus) atomicAdd(m_rari + roff, rval);
		else *(m_rari + roff) = rval;
		break;
	case AOP_PLUS:
		rval = *(m_pari + poff);
		if(rplus) atomicAdd(m_rari + roff, rval);
		else *(m_rari + roff) = rval;
		break;
	case AOP_DIV:
		break;
	case AOP_MINUS:
		break;
	case ABP_MINUS_PREF:
		rval = *(m_pari + poff);
		if(rplus) atomicAdd(m_rari + roff, rval);
		else *(m_rari + roff) = rval;
		break;
	case ABP_MINUS_SUFF:
		rval = *(m_pari + poff) * -1;
		if(rplus) atomicAdd(m_rari + roff, rval);
		else *(m_rari + roff) = rval;
		break;
	case ABP_DIV_PREF:
		rval = *(m_pari + poff) * (1 / *(m_sari + poff % ssz));
		if(rplus) atomicAdd(m_rari + roff, rval);
		else *(m_rari + roff) = rval;
		break;
	case ABP_DIV_SUFF:
		rval = *(m_pari + poff) * (1 / (*(m_sari + poff % ssz) * *(m_sari + poff % ssz)) * -1);
		if(rplus) atomicAdd(m_rari + roff, rval);
		else *(m_rari + roff) = rval;
		break;
	case ABP_BWTEST:
		rval = *(m_pari + poff) / *(m_sari + poff % ssz);
		if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
		else if(*(m_rari + roff) != rval) printf("xxx\n");
		break;
	}
}
template<typename T>
__global__ void karith_f2(void *pcxt, T *m_pari, T *m_sari, T *m_rari, intt idx_origin, intt idx_width, sytet aop, T rplus, intt n)
{//어느 한쪽이 작은 브로드 케스트(디멘젼이 1인 차원이 없는)
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	ArithVar *arv = (ArithVar *)pcxt;
	intt osz = arv->zarOut, ssz = arv->zarSuf, psz = arv->zarPre, poff;
	T rval;
	if(arv->bwGetOri) {//pre와 out의 사이즈가 같은 경우, pre사이즈가 out보다 큰 경우는 위 함수에서 수행, pre는 순전파에서
		switch(aop) {	//리턴 매트릭스이므로 pre가 더 적은 경우는 없음.
		case AOP_MUL:
			for(poff = roff;poff < psz; poff += osz) {
				rval = *(m_pari + poff) * *(m_sari + poff % ssz);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case AOP_PLUS:
			for(poff = roff;poff < psz; poff += osz) {
				rval = *(m_pari + poff);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case AOP_DIV:
			break;
		case AOP_MINUS:
			break;
		case ABP_MINUS_PREF:
			for(poff = roff;poff < psz; poff += osz) {
				rval = *(m_pari + poff);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_MINUS_SUFF:
			for(poff = roff;poff < psz; poff += osz) {
				rval = *(m_pari + poff) * -1;
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_DIV_PREF:
			for(poff = roff;poff < psz; poff += osz) {
				rval = *(m_pari + poff) * (1 / *(m_sari + poff % ssz));
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_DIV_SUFF:
			for(poff = roff;poff < psz; poff += osz) {
				rval = *(m_pari + poff) * (1 / (*(m_sari + poff % ssz) * *(m_sari + poff % ssz)) * -1);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_BWTEST:
			for(poff = roff;poff < psz; poff += osz) {
				rval = *(m_pari + poff) / *(m_sari + poff % ssz);
				if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
				else if(*(m_rari + roff) != rval) printf("xxx\n");
			}
			break;
		}
	} else {
		switch(aop) {
		case AOP_MUL:
			rval = *(m_pari + roff % psz) * *(m_sari + roff % ssz);
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_PLUS:
			rval = *(m_pari + roff % psz) + *(m_sari + roff % ssz);
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_DIV:
			rval = *(m_pari + roff % psz) / *(m_sari + roff % ssz);
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_MINUS:
			rval = *(m_pari + roff % psz) - *(m_sari + roff % ssz);
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		}
	}
}
template<typename T>
__global__ void karith_f2_lc(void *pcxt, T *m_sari, T *m_rari, intt idx_origin, intt idx_width, T sval, sytet aop, T rplus, intt n)
{//좌측이 상수인 브로드 케스트(디멘젼이 1인 차원이 없는)
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	ArithVar *arv = (ArithVar *)pcxt;
	T rval;
	if(arv->bwGetOri) {
		switch(aop) {
		case AOP_MUL:
			rval = (sval * *(m_sari + roff));
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case AOP_PLUS:
			if(rplus) *(m_rari + roff) += sval;
			else *(m_rari + roff) = sval;
			break;
		case AOP_DIV:
			break;
		case AOP_MINUS:
			break;
		case ABP_MINUS_PREF:
			if(rplus) *(m_rari + roff) += sval;
			else *(m_rari + roff) = sval;
			break;
		case ABP_MINUS_SUFF:
			if(rplus) *(m_rari + roff) += sval * -1;
			else *(m_rari + roff) = sval * -1;
			break;
		case ABP_DIV_PREF:
			rval = sval * (1 / *(m_sari + roff));
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case ABP_DIV_SUFF:
			rval = sval * (1 / (*(m_sari + roff) * *(m_sari + roff)) * -1);
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case ABP_BWTEST:
			rval = sval / *(m_sari + roff);
			if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
			else if(*(m_rari + roff) != rval)  printf("xxx\n");
			break;
		}
	} else {
		switch(aop) {
		case AOP_MUL:
			rval = (sval * *(m_sari + roff));
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_PLUS:
			rval = (sval + *(m_sari + roff));
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_DIV:
			rval = (sval / *(m_sari + roff));
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_MINUS:
			rval = (sval - *(m_sari + roff));
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		}
	}
}
template<typename T>
__global__ void karith_f2_rc(void *pcxt, T *m_pari, T *m_rari, intt idx_origin, intt idx_width, T sval, sytet aop, T rplus, intt n)
{//우측이 상수인 브로드 케스트(디멘젼이 1인 차원이 없는)
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	ArithVar *arv = (ArithVar *)pcxt;
	T rval;
	if(arv->bwGetOri) {
		switch(aop) {
		case AOP_MUL:
			rval = (*(m_pari + roff) * sval);
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case AOP_PLUS:
			rval = *(m_pari + roff);
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case AOP_DIV:
			break;
		case AOP_MINUS:
			break;
		case ABP_MINUS_PREF:
			rval = *(m_pari + roff);
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case ABP_MINUS_SUFF:
			rval = *(m_pari + roff) * -1;
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case ABP_DIV_PREF:
			rval = *(m_pari + roff) * (1 / sval);
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case ABP_DIV_SUFF:
			rval = *(m_pari + roff) * (1 / (sval * sval) * -1);
			if(rplus) *(m_rari + roff) += rval;
			else *(m_rari + roff) = rval;
			break;
		case ABP_BWTEST:
			rval = *(m_pari + roff) / sval;
			if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
			else if(*(m_rari + roff) != rval)  printf("xxx\n");
			break;
		}
	} else {
		switch(aop) {
		case AOP_MUL:
			rval = *(m_pari + roff) * sval;
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_PLUS:
			rval = *(m_pari + roff) + sval;
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_DIV:
			rval = *(m_pari + roff) / sval;
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_MINUS:
			rval = *(m_pari + roff) - sval;
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		}
	}
}
template<typename T>
__global__ void karith_f3(void *pcxt, T *m_pari, T *m_sari, T *m_rari, intt idx_origin, intt idx_width, sytet aop, T rplus, intt n)
{//디멘젼이 1인 차원을 포함하는 브로드 케스트, 어느 한쪽이 상수이면 상대측 디멘젼이 1인 차원이 의미가 없어지므로 타입 2번 케이스가 실행됨.
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	ArithVar *arv = (ArithVar *)pcxt;
	intt *mrank = arv->arRankMast, *prank = arv->arRankPre, *srank = arv->arRankSuf, *rrank = arv->arRankRet;
	intt npre = arv->narPre, nsuf = arv->narSuf, nmast = arv->narMast, nret = arv->narRet;
	intt cidx[MX_DIM], tmp_idx[MX_DIM], poff, soff;
	T rval;
	bool end_check;

	if(arv->bwGetOri) {
		_lead_offset2idx(arv->narBro, npre, nret, rrank, roff, cidx, end_check);
		switch(aop) {
		case AOP_MUL:
			for(;;) {
				_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
				_moff2soff(nmast, mrank, nsuf, srank, poff, tmp_idx, soff);
				rval = *(m_pari + poff) * *(m_sari + soff);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case AOP_PLUS:
			for(;;) {
				_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
				rval = *(m_pari + poff);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case AOP_DIV:
			break;
		case AOP_MINUS:
			break;
		case ABP_MINUS_PREF:
			for(;;) {
				_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
				rval = *(m_pari + poff);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_MINUS_SUFF:
			for(;;) {
				_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
				rval = *(m_pari + poff) * -1;
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_DIV_PREF:
			for(;;) {
				_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
				_moff2soff(nmast, mrank, nsuf, srank, poff, tmp_idx, soff);
				rval = *(m_pari + poff) * (1 / *(m_sari + soff));
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_DIV_SUFF:
			for(;;) {
				_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
				_moff2soff(nmast, mrank, nsuf, srank, poff, tmp_idx, soff);
				rval = *(m_pari + poff) * (1 / (*(m_sari + soff) * *(m_sari + soff)) * -1);
				if(rplus) *(m_rari + roff) += rval;
				else *(m_rari + roff) = rval;
			}
			break;
		case ABP_BWTEST:
			for(;;) {
				_bro_offset(arv->narBro, arv->broDimRet, arv->broIdxRet, nmast, mrank, cidx, poff, end_check);
				_moff2soff(nmast, mrank, nsuf, srank, poff, tmp_idx, soff);
				rval = *(m_pari + poff) / *(m_sari + soff);
				if(*(m_rari + roff) == 0) *(m_rari + roff) = rval;
				else if(*(m_rari + roff) != rval)  printf("xxx\n");
			}
			break;
		}
	} else {
		_moff2soff(nmast, mrank, npre, prank, roff, tmp_idx, poff);
		_moff2soff(nmast, mrank, nsuf, srank, roff, tmp_idx, soff);
		//printf("11111111 %d %d %f\n", poff, soff, rval);
		switch(aop) {
		case AOP_MUL:
			rval = *(m_pari + poff) * *(m_sari + soff);
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_PLUS:
			rval = *(m_pari + poff) + *(m_sari + soff);
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_DIV:
			rval = *(m_pari + poff) / *(m_sari + soff);
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		case AOP_MINUS:
			rval = *(m_pari + poff) - *(m_sari + soff);
			if(rplus) {
				*(m_rari + roff) *= rplus;
				*(m_rari + roff) += rval;
			} else *(m_rari + roff) = rval;
			break;
		}
	}
}
template<typename T>
intt garith_t(void *pcxt, T *m_pari, T *m_sari, T *m_rari, intt p_size, intt r_size,
	intt idx_origin, intt idx_width, T sval, sytet aop, T rplus, sytet tp_arith)
{
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);
	if(tp_arith == AR_T_BRO && p_size > r_size) {//pre사이즈가 ret사이즈보다 큰경우는 브로드케스트 역전파 경우밖에 없고.
		dim3 block(WIDTH_BLOCK3(p_size));			//이경우 pre를 기준으로 루프를 구성한다.
		dim3 grid((p_size + block.x - 1) / block.x);
		karith_f2_bwprem<T> << <grid, block >> > (pcxt, m_pari, m_sari, m_rari, idx_origin, idx_width, aop, rplus, n);
	} else {
		intt bsz = (tp_arith == AR_T_ONEBRO ? SMALL_BLOCK : BLOCK_SIZE);
		dim3 block(WIDTH_BLOCK2(bsz));
		dim3 grid((idx_width + block.x - 1) / block.x);
		//karith_f<T> << <grid, block >> > (pcxt, m_pari, m_sari, m_rari, idx_origin, idx_width, sval, aop, rplus, n);
		switch(tp_arith) {
		case AR_T_O2O:
			karith_f1<T> << <grid, block >> > (pcxt, m_pari, m_sari, m_rari, idx_origin, idx_width, aop, rplus, n);
			break;
		case AR_T_BRO:
			karith_f2<T> << <grid, block >> > (pcxt, m_pari, m_sari, m_rari, idx_origin, idx_width, aop, rplus, n);
			break;
		case AR_T_BROLC:
			karith_f2_lc<T> << <grid, block >> > (pcxt, m_sari, m_rari, idx_origin, idx_width, sval, aop, rplus, n);
			break;
		case AR_T_BRORC:
			karith_f2_rc<T> << <grid, block >> > (pcxt, m_pari, m_rari, idx_origin, idx_width, sval, aop, rplus, n);
			break;
		case AR_T_ONEBRO:
			karith_f3<T> << <grid, block >> > (pcxt, m_pari, m_sari, m_rari, idx_origin, idx_width, aop, rplus, n);
			break;
		}
	}
	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt garith_f(void *pcxt, floatt *m_pari, floatt *m_sari, floatt *m_rari, intt p_size, intt r_size,
	intt idx_origin, intt idx_width, floatt sval, sytet aop, floatt rplus, sytet tp_arith)
{
	return garith_t<floatt>(pcxt, m_pari, m_sari, m_rari, p_size, r_size,
		idx_origin, idx_width, sval, aop, rplus, tp_arith);
}
intt garith_f(void *pcxt, intt *m_pari, intt *m_sari, intt *m_rari, intt p_size, intt r_size,
	intt idx_origin, intt idx_width, intt sval, sytet aop, intt rplus, sytet tp_arith)
{
	return garith_t<intt>(pcxt, m_pari, m_sari, m_rari, p_size, r_size,
		idx_origin, idx_width, sval, aop, rplus, tp_arith);
}

template<typename T>
__global__ void ktrans_f(void *pcxt, T *m_strs, T *m_rtrs, intt idx_origin, intt idx_width, bool bw, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	TransVar *tsvar = (TransVar *)pcxt;
	intt ndims = tsvar->ntrDims, *rrank = tsvar->trRankRet, i, ridx[MX_DIM];
	TransRank *tmap = tsvar->tspmap;

	_offset2idx(ndims, rrank, roff, ridx);
	for(i = 0;i < ndims; i++) {
		m_strs += (ridx[i] * tmap[i].trssz);
	}
	if(bw) *(m_rtrs + roff) += *m_strs;
	else *(m_rtrs + roff) = *m_strs;
}
template<typename T>
intt gtrans_t(void *pcxt, T *m_strs, T *m_rtrs, intt r_size, intt idx_origin, intt idx_width, bool bw)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	ktrans_f<T> << <grid, block >> > (pcxt, m_strs, m_rtrs, idx_origin, idx_width, bw, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gtrans_f(void *pcxt, floatt *m_strs, floatt *m_rtrs, intt r_size, intt idx_origin, intt idx_width, bool bw)
{
	return gtrans_t<floatt>(pcxt, m_strs, m_rtrs, r_size, idx_origin, idx_width, bw);
}
intt gtrans_f(void *pcxt, intt *m_strs, intt *m_rtrs, intt r_size, intt idx_origin, intt idx_width, bool bw)
{
	return gtrans_t<intt>(pcxt, m_strs, m_rtrs, r_size, idx_origin, idx_width, bw);
}
template<typename T>
__device__ void datomic_max_f(T *pmax, const T value)
{
	if(*pmax >= value) return;

	intt * const imax = (int *)pmax;
	intt old = *imax, vcmp;

	do
	{
		vcmp = old;
		if(__int_as_float(vcmp) >= value) break;

		old = atomicCAS(imax, vcmp, __float_as_int(value));
	} while(vcmp != old);
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

template<typename T>
__global__ void ksoftx_cross_e_f(void *pcxt, T *m_ssfx, T *m_rsfx, T *m_tsfx, intt f_size, intt idx_origin, intt idx_width, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	atomicAdd(m_rsfx + (roff / f_size), -1.0f * std::log(*(m_ssfx + roff) + 1e-8) * *(m_tsfx + roff));
	//printf("%f %f %f\n", *(m_rsfx + (roff / f_size)), *(m_ssfx + roff), *(m_tsfx + roff));
}
template<typename T>
intt gsoftx_cross_e_t(void *pcxt, T *m_ssfx, T *m_rsfx, T *m_tsfx, intt r_size, intt f_size, intt idx_origin, intt idx_width)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	ksoftx_cross_e_f<T> << <grid, block >> > (pcxt, m_ssfx, m_rsfx, m_tsfx, f_size, idx_origin, idx_width, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gsoftx_cross_e_f(void *pcxt, floatt *m_ssfx, floatt *m_rsfx, floatt *m_tsfx, intt r_size, intt f_size, intt idx_origin, intt idx_width)
{
	return gsoftx_cross_e_t<floatt>(pcxt, m_ssfx, m_rsfx, m_tsfx, r_size, f_size, idx_origin, idx_width);
}
intt gsoftx_cross_e_f(void *pcxt, intt *m_ssfx, intt *m_rsfx, intt *m_tsfx, intt r_size, intt f_size, intt idx_origin, intt idx_width)
{
	return gsoftx_cross_e_t<intt>(pcxt, m_ssfx, m_rsfx, m_tsfx, r_size, f_size, idx_origin, idx_width);
}

template<typename T>
__global__ void ksum_f(void *pcxt, T *m_smet, T *m_rmet, intt idx_origin, intt idx_width, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	atomicAdd(m_rmet, *(m_smet + roff));
}
template<typename T>
__global__ void kmean_f(T *m_rmet, T *cmul, bool mean, intt r_size)
{
	if(cmul) *m_rmet *= *(T *)cmul;
	if(mean) *m_rmet /= r_size;
	//printf("%p %d %f\n", cmul, mean, *m_rmet);
}
template<typename T>
intt gsum_t(void *pcxt, T *m_smet, T *m_rmet, intt r_size, intt idx_origin, intt idx_width, T *cmul, bool mean)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	ksum_f<T> << <grid, block >> > (pcxt, m_smet, m_rmet, idx_origin, idx_width, n);
	if(n == r_size && (cmul || mean)) kmean_f<T> << <1, 1 >> > (m_rmet, cmul, mean, r_size);//마지막 분할에서 평균구하기 실행.

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gsum_f(void *pcxt, floatt *m_smet, floatt *m_rmet, intt r_size, intt idx_origin, intt idx_width, floatt *cmul, bool mean)
{
	return gsum_t<floatt>(pcxt, m_smet, m_rmet, r_size, idx_origin, idx_width, cmul, mean);
}
intt gsum_f(void *pcxt, intt *m_smet, intt *m_rmet, intt r_size, intt idx_origin, intt idx_width, intt *cmul, bool mean)
{
	return gsum_t<intt>(pcxt, m_smet, m_rmet, r_size, idx_origin, idx_width, cmul, mean);
}

template<typename T>
__global__ void koptadm_f(T *mm, T *mv, T *mg, T *mr, intt idx_origin,
	intt idx_width, T beta1, T beta2, T lr, T ep, intt dec, sytet db, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	mm[roff] += (1.0f - beta1) * (mg[roff] - mm[roff]);//우항: mm[roff] + (l - beta1)*mg[roff] - mm[roff] + mm[roff]*beta1
													//	  = mm[roff]*beta1 + (l - beta1)*mg[roff]
	mv[roff] += (1.0f - beta2) * (mg[roff] * mg[roff] - mv[roff]);
	mr[roff] += dec * lr * mm[roff] / (std::sqrt(db ? (doublet)mv[roff] : (floatt)mv[roff]) + ep);
}
template<typename T>
intt goptadm_t(void *pcxt, T *mm, T *mv, T *mg, T *mr, intt r_size, intt idx_origin,
	intt idx_width, T beta1, T beta2, T lr, T ep, intt dec, sytet db)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	koptadm_f<T> << <grid, block >> > (mm, mv, mg, mr, idx_origin, idx_width, beta1, beta2, lr, ep, dec, db, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt goptadm_f(void *pcxt, floatt *mm, floatt *mv, floatt *mg, floatt *mr, intt r_size, intt idx_origin,
	intt idx_width, floatt beta1, floatt beta2, floatt lr, floatt ep, intt dec, sytet db)
{
	return goptadm_t<floatt>(pcxt, mm, mv, mg, mr, r_size, idx_origin,
		idx_width, beta1, beta2, lr, ep, dec, db);
}
intt goptadm_f(void *pcxt, intt *mm, intt *mv, intt *mg, intt *mr, intt r_size, intt idx_origin,
	intt idx_width, intt beta1, intt beta2, intt lr, intt ep, intt dec, sytet db)
{
	return goptadm_t<intt>(pcxt, mm, mv, mg, mr, r_size, idx_origin,
		idx_width, beta1, beta2, lr, ep, dec, db);
}

template<typename T>
__global__ void koptsgd_f(T *mg, T *mr, intt idx_origin, intt idx_width, T lr, intt dec, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	mr[roff] += dec * lr * mg[roff];
}
template<typename T>
intt goptsgd_t(void *pcxt, T *mg, T *mr, intt r_size, intt idx_origin, intt idx_width, T lr, intt dec)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	koptsgd_f<T> << <grid, block >> > (mg, mr, idx_origin, idx_width, lr, dec, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt goptsgd_f(void *pcxt, floatt *mg, floatt *mr, intt r_size, intt idx_origin, intt idx_width, floatt lr, intt dec)
{
	return goptsgd_t<floatt>(pcxt, mg, mr, r_size, idx_origin, idx_width, lr, dec);
}
intt goptsgd_f(void *pcxt, intt *mg, intt *mr, intt r_size, intt idx_origin, intt idx_width, intt lr, intt dec)
{
	return goptsgd_t<intt>(pcxt, mg, mr, r_size, idx_origin, idx_width, lr, dec);
}
template<typename T>
__device__ __forceinline__ T mat_sqrt(T a, sytet db)
{
	return std::sqrt(db ? (doublet)a : (floatt)a);
}
template<typename T>
__device__ __forceinline__ T mat_exp(T a, sytet db)
{
	return std::exp(db ? (doublet)a : (floatt)a);
}
template<typename T>
__device__ __forceinline__ float mat_log(T a, sytet db) {
	return std::log(db ? (doublet)a : (floatt)a);
}
template<typename T>
__global__ void kactf_f(T *mpre, T *mret, intt idx_origin, intt idx_width, intt aop2, sytet rplus, sytet db, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;
	T v;

	if(roff >= n) return;

	switch(aop2) {
	case ACTF_TANH:
		mret[roff] = std::tanh(db ? (doublet)mpre[roff] : (floatt)mpre[roff]);
		break;
	case DACTF_TANH:
		if(rplus) mret[roff] += (1.0f - std::tanh(db ? (doublet)mpre[roff] : (floatt)mpre[roff]) *
					std::tanh(db ? (doublet)mpre[roff] : (floatt)mpre[roff]));
		else mret[roff] = 1.0f - std::tanh(db ? (doublet)mpre[roff] : (floatt)mpre[roff]) * 
					std::tanh(db ? (doublet)mpre[roff] : (floatt)mpre[roff]);
		break;
	case ACTF_RELU:
		mret[roff] = mpre[roff] > 0.0f ? mpre[roff] : 0.0f;
		break;
	case DACTF_RELU:
		if(rplus) mret[roff] += (mpre[roff] > 0.0f ? 1.0f : 0.0f);
		else mret[roff] = mpre[roff] > 0.0f ? 1.0f : 0.0f;
		break;
	case ACTF_SIGM:
		mret[roff] = 1.0 / (1.0 + mat_exp(-mpre[roff], db));//1.0f/(1.0f + std::exp(-a));
		break;
	case DACTF_SIGM:
		v = 1.0 / (1.0 + mat_exp(-mpre[roff], db));
		if(rplus) mret[roff] += (1.0 - v) * v;
		else mret[roff] = (1.0 - v) * v;
		break;
	case MATH_SQRT:
		mret[roff] = mat_sqrt(mpre[roff], db);
		break;
	case DMATH_SQRT:
		if(rplus) mret[roff] += 0.5 * 1.0 / mat_sqrt(mpre[roff], db);//0.5 * pow(mpre[roff], -0.5f)
		else mret[roff] = 0.5 * 1.0 / mat_sqrt(mpre[roff], db);//0.5 * pow(mpre[roff], -0.5f)
		break;
	case JUST_COPY:
		break;
	case DJUST_COPY://단순하 역전파때 기울기를 더하기위해 사용.
		if(rplus) mret[roff] += mpre[roff];
		else mret[roff] = mpre[roff];
		break;
	case MATH_LOG:
		mret[roff] = mat_log(mpre[roff], db);
		break;
	}
}
template<typename T>
intt gactf_t(void *pcxt, T *mpre, T *mret, intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet rplus, sytet db)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	kactf_f<T> << <grid, block >> > (mpre, mret, idx_origin, idx_width, aop2, rplus, db, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gactf_f(void *pcxt, floatt *mpre, floatt *mret, intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet rplus, sytet db)
{
	return gactf_t<floatt>(pcxt, mpre, mret, r_size, idx_origin, idx_width, aop2, rplus, db);
}
intt gactf_f(void *pcxt, intt *mpre, intt *mret, intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet rplus, sytet db)
{
	return gactf_t<intt>(pcxt, mpre, mret, r_size, idx_origin, idx_width, aop2, rplus, db);
}
template<typename T>
__global__ void kactf2_f(T *mpre, T *msuf, T *mret, T *rsuf, intt idx_origin, intt idx_width, intt aop2, sytet rplus, sytet db, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	switch(aop2) {
	case ACTF_PRELU:
		mret[roff] = mpre[roff] > 0.0 ? mpre[roff] : msuf[roff] * mpre[roff];// x > 0.0f ? x : a * x
		break;
	case DACTF_PRELU:
		if(rplus) {
			mret[roff] += (mpre[roff] > 0.0 ? 1.0 : msuf[roff]);//dst[idx] = src[idx] > 0.0f ? 1.0f : a[idx];
			rsuf[roff] += (mpre[roff] > 0.0 ? 0.0 : mpre[roff]);//da[idx] = src[idx] > 0.0f ? 0.0f : src[idx];
		} else {
			mret[roff] = mpre[roff] > 0.0 ? 1.0 : msuf[roff];//dst[idx] = src[idx] > 0.0f ? 1.0f : a[idx];
			rsuf[roff] = mpre[roff] > 0.0 ? 0.0 : mpre[roff];//da[idx] = src[idx] > 0.0f ? 0.0f : src[idx];
		}
		break;
	}
}
template<typename T>
intt gactf2_t(void *pcxt, T *mpre, T *msuf, T *mret, T *rsuf, intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet rplus, sytet db)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	kactf2_f<T> << <grid, block >> > (mpre, msuf, mret, rsuf, idx_origin, idx_width, aop2, rplus, db, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gactf2_f(void *pcxt, floatt *mpre, floatt *msuf, floatt *mret, floatt *rsuf, intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet rplus, sytet db)
{
	return gactf2_t<floatt>(pcxt, mpre, msuf, mret, rsuf, r_size, idx_origin, idx_width, aop2, rplus, db);
}
intt gactf2_f(void *pcxt, intt *mpre, intt *msuf, intt *mret, intt *rsuf, intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet rplus, sytet db)
{
	return gactf2_t<intt>(pcxt, mpre, msuf, mret, rsuf, r_size, idx_origin, idx_width, aop2, rplus, db);
}
template<typename T>
__global__ void kembedding_f(T *mpre, T *msuf, T *mret, intt idx_origin, intt idx_width, intt sz_embed, intt stp, intt bw, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;
	intt idx;

	if(roff >= n) return;

	int_val_type(idx, &msuf[roff / sz_embed], stp);
	if(bw) {//msuf - input, mret - lookup table, mpre - embeded, roff는 mpre 기준, roff가 mret기준이 아니므로
			//쓰레드(cpu쓰레드뿐 아니라 gpu 커널 쓰레드도)간 mret에 결과가 중첩될수있어 배타처리 한다.
		atomicAdd(&mret[idx*sz_embed + roff % sz_embed], mpre[roff]);
	} else {//msuf - input, mret - embeded, mpre - lookup table, roff는 mret 기준
		mret[roff] = mpre[idx*sz_embed + roff % sz_embed];
	}
}
template<typename T>
intt gembedding_t(T *mpre, T *msuf, T *mret, intt r_size, intt idx_origin, intt idx_width, intt sz_embed, intt stp, intt bw)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	kembedding_f<T> << <grid, block >> > (mpre, msuf, mret, idx_origin, idx_width, sz_embed, stp, bw, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gembedding_f(floatt *mpre, floatt *msuf, floatt *mret, intt r_size, intt idx_origin, intt idx_width, intt sz_embed, intt stp, intt bw)
{
	return gembedding_t<floatt>(mpre, msuf, mret, r_size, idx_origin, idx_width, sz_embed, stp, bw);
}
intt gembedding_f(intt *mpre, intt *msuf, intt *mret, intt r_size, intt idx_origin, intt idx_width, intt sz_embed, intt stp, intt bw)
{
	return gembedding_t<intt>(mpre, msuf, mret, r_size, idx_origin, idx_width, sz_embed, stp, bw);
}
template<typename T>
__global__ void konehot_f(void *pcxt, T *mpre, T *mret, intt idx_origin, intt idx_width, intt n)
{
	OneVar *ovar = (OneVar *)pcxt;
	intt poff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x, depth = ovar->idxOne[5];

	if(poff >= n || *(mpre + poff) >= depth || *(mpre + poff) < 0) return;
	
	intt roff, psz = ovar->idxOne[4];

	roff = (poff / psz) * depth * psz + *(mpre + poff) * psz + poff % psz;
	*(mret + roff) = (T)*(doublet *)ovar->idxOne;
}
template<typename T>
intt gonehot_t(void *pcxt, T *mpre, T *mret, intt r_size, intt idx_origin, intt idx_width)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	konehot_f<T> << <grid, block >> > (pcxt, mpre, mret, idx_origin, idx_width, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gonehot_f(void *pcxt, floatt *mpre, floatt *mret, intt r_size, intt idx_origin, intt idx_width)
{
	return gonehot_t<floatt>(pcxt, mpre, mret, r_size, idx_origin, idx_width);
}
intt gonehot_f(void *pcxt, intt *mpre, intt *mret, intt r_size, intt idx_origin, intt idx_width)
{
	return gonehot_t<intt>(pcxt, mpre, mret, r_size, idx_origin, idx_width);
}
template<typename T>
__global__ void kslice_f(void *pcxt, T *mpre, T *mret, intt idx_origin, intt idx_width, bool bw, sytet rplus, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	OneVar *ovar = (OneVar *)pcxt;
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
	if(bw) {
		if(rplus) *cmem += *(smem + roff);
		else *cmem = *(smem + roff);
	} else *(smem + roff) = *cmem;
}
template<typename T>
intt gslice_t(void *pcxt, T *mpre, T *mret, intt r_size, intt idx_origin, intt idx_width, bool bw, sytet rplus)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	kslice_f<T> << <grid, block >> > (pcxt, mpre, mret, idx_origin, idx_width, bw, rplus, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gslice_f(void *pcxt, floatt *mpre, floatt *mret, intt r_size, intt idx_origin, intt idx_width, bool bw, sytet rplus)
{
	return gslice_t<floatt>(pcxt, mpre, mret, r_size, idx_origin, idx_width, bw, rplus);
}
intt gslice_f(void *pcxt, intt *mpre, intt *mret, intt r_size, intt idx_origin, intt idx_width, bool bw, sytet rplus)
{
	return gslice_t<intt>(pcxt, mpre, mret, r_size, idx_origin, idx_width, bw, rplus);
}
template<typename T>
__global__ void kargmax_f(void *pcxt, T *mpre, T *mret, intt idx_origin, intt idx_width, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	OneVar *ovar = (OneVar *)pcxt;
	intt poff, i, naxis = ovar->nrkPre, inner_sz = ovar->nrkSuf, outer_sz = ovar->nrkOut;
	T vmax;

	for(;roff < n; roff++) {
		poff = (roff / inner_sz) * outer_sz + roff % inner_sz;
		for(i = 0, vmax = 0;i < naxis; i++, poff += inner_sz) {
			if(vmax < *(mpre + poff)) {
				vmax = *(mpre + poff);
				*(mret + roff) = i;
			}
		}
	}
}
template<typename T>
intt gargmax_t(void *pcxt, T *mpre, T *mret, intt r_size, intt idx_origin, intt idx_width)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	kargmax_f<T> << <grid, block >> > (pcxt, mpre, mret, idx_origin, idx_width, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gargmax_f(void *pcxt, floatt *mpre, floatt *mret, intt r_size, intt idx_origin, intt idx_width)
{
	return gargmax_t<floatt>(pcxt, mpre, mret, r_size, idx_origin, idx_width);
}
intt gargmax_f(void *pcxt, intt *mpre, intt *mret, intt r_size, intt idx_origin, intt idx_width)
{
	return gargmax_t<intt>(pcxt, mpre, mret, r_size, idx_origin, idx_width);
}

template<typename T>
__global__ void kequal_f(void *pcxt, T *mpre, T *msuf, T *mret, intt idx_origin, intt idx_width, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	OneVar *ovar = (OneVar *)pcxt;
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
}
template<typename T>
intt gequal_t(void *pcxt, T *mpre, T *msuf, T *mret, intt r_size, intt idx_origin, intt idx_width)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	kequal_f<T> << <grid, block >> > (pcxt, mpre, msuf, mret, idx_origin, idx_width, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gequal_f(void *pcxt, floatt *mpre, floatt *msuf, floatt *mret, intt r_size, intt idx_origin, intt idx_width)
{
	return gequal_t<floatt>(pcxt, mpre, msuf, mret, r_size, idx_origin, idx_width);
}
intt gequal_f(void *pcxt, intt *mpre, intt *msuf, intt *mret, intt r_size, intt idx_origin, intt idx_width)
{
	return gequal_t<intt>(pcxt, mpre, msuf, mret, r_size, idx_origin, idx_width);
}
template<typename T>
__global__ void ktype1_t(void *pcxt, T *mpre, T *msuf, T *mret, intt idx_origin, intt idx_width, intt aop2, intt n)
{
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;
	
	if(roff >= n) return;

	OneVar *ovar = (OneVar *)pcxt;
	doublet low = *(doublet *)&ovar->idxOne[0], high = *(doublet *)&ovar->idxOne[2];

	switch(aop2) {
	case TYPE1_CLIP:
		if(mpre[roff] < low) mret[roff] = low;
		else if(mpre[roff] > high) mret[roff] = high;
		else mret[roff] = mpre[roff];
		break;
	}
}
template<typename T>
intt gtype1_t(void *pcxt, T *mpre, T *msuf, T *mret, intt r_size, intt idx_origin, intt idx_width, intt aop2)
{
	dim3 block(WIDTH_BLOCK);
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (r_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : r_size);

	ktype1_t<T> << <grid, block >> > (pcxt, mpre, msuf, mret, idx_origin, idx_width, aop2, n);

	cudaThreadSynchronize();
	return n - idx_origin * idx_width;
}
intt gtype1_f(void *pcxt, floatt *mpre, floatt *msuf, floatt *mret, intt r_size, intt idx_origin, intt idx_width, intt aop2)
{
	return gtype1_t<floatt>(pcxt, mpre, msuf, mret, r_size, idx_origin, idx_width, aop2);
}
intt gtype1_f(void *pcxt, intt *mpre, intt *msuf, intt *mret, intt r_size, intt idx_origin, intt idx_width, intt aop2)
{
	return gtype1_t<intt>(pcxt, mpre, msuf, mret, r_size, idx_origin, idx_width, aop2);
}
#include <curand_kernel.h>
template<typename T>
__global__ void knormal_t(T *mpre, curandState *cust, intt n)
{
	intt roff = blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	mpre[roff] = curand_normal(&cust[roff]);
}
template<typename T>
__global__ void kuniform_t(T *mpre, curandState *cust, intt n)
{
	intt roff = blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	mpre[roff] = curand_uniform(&cust[roff]);
}
__global__ void seed_random(curandState *cus, intt seed, intt n)
{
	intt roff = blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	curand_init(seed, roff, 0, &cus[roff]);
}
template<typename T>
intt grandom_t(T *mpre, intt idx_width, intt aop2)
{
	dim3 block(WIDTH_BLOCK2(SMALL_BLOCK));
	dim3 grid((idx_width + block.x - 1) / block.x);

	curandState *cust;
	cudaMalloc((void **)&cust, idx_width * sizeof(curandState));
	seed_random<< <grid, block >> > (cust, 0, idx_width);

	switch(aop2) {
	case RAND_T_N:
		knormal_t<T> << <grid, block >> > (mpre, cust, idx_width);
		break;
	case RAND_T_U:
		kuniform_t<T> << <grid, block >> > (mpre, cust, idx_width);
		break;
	case RAND_T_L:
		break;
	case RAND_T_P:
		break;
	}
	cudaFree(cust);
	cudaThreadSynchronize();
	return idx_width;
}
intt grandom_f(floatt *mpre, intt r_size, intt aop2)
{
	return grandom_t<floatt>(mpre, r_size, aop2);
}
intt grandom_f(intt *mpre, intt r_size, intt aop2)
{
	return grandom_t<intt>(mpre, r_size, aop2);
}