
#include "mgpu.h"
#include "matrix.h"
#include <stdio.h>

/*//dot version 1
template<typename T>
__global__ void kdot_f(void *pcxt, T *m_pdot, T *m_sdot, T *m_rdot, intt idx_origin, intt idx_width, T rplus, intt n)
{
	DotVar *dotv = (DotVar *)pcxt;
	intt roff = idx_width * idx_origin + blockIdx.x*blockDim.x + threadIdx.x;

	if(roff >= n) return;

	intt pidx[MX_DIM], sidx[MX_DIM], ridx[MX_DIM], tmp_idx[MX_DIM], poff, soff;
	intt sz_suf_shrink = dotv->szShrinkSuf, jo_top_axis_p = dotv->joTopAxisPre, jo_top_axis_s = dotv->joTopAxisSuf;
	intt pdim = dotv->ndimPre, sdim = dotv->ndimSuf, n_joint_axis = dotv->nJointAxis;
	intt *prank = dotv->szRankPre, *srank = dotv->szRankSuf, i, j;
	T sum = 0;

	if(dotv->bwGetOri == BWDIV_PREF) {//A*B=C, A(ret)=C*B',순전파때 A의 출력축은 역전파때 C의 출력축이되고 순전파때 A의 조인축은
		doffset2idx(dotv->ndimRet, dotv->szRankRet, roff, ridx);//역전파때 B의 출력축이된다.여기서 B외C의 출력축을 정하고 밑에서
		if(dotv->noutPre > 1 || dotv->noutRet > 1) {			//조인축을 따라 조인
			soff = dsparse_idx2offset(dotv->noutRet, dotv->outRankRet, ridx, dotv->outAxisRet);//A의 출력랭크로 옵셋변환후
			doffset2idx(dotv->noutPre, dotv->outRankPre, soff, tmp_idx);//변환 옵셋과 C의 출력 랭크로 인덱스 게산
			for(j = 0;j < dotv->noutPre; j++) pidx[dotv->outAxisPre[j]] = tmp_idx[j];//C의 출력 인덱스 설정
		} else {//A의 출력 인덱스를 C의 출력 인덱스에 바로 설정.
			for(j = 0;j < dotv->noutPre; j++) pidx[dotv->outAxisPre[j]] = ridx[dotv->outAxisRet[j]];//전체 인덱스에 출력차원 인덱스 설정.
		}
		if(dotv->noutSuf > 1 || dotv->njoRet > 1) {
			soff = dsparse_idx2offset(dotv->njoRet, dotv->joRankRet, ridx, dotv->joAxisRet);//A의 조인랭크로 옵셋변환후
			doffset2idx(dotv->noutSuf, dotv->outRankSuf, soff, tmp_idx);//변환 옵셋과 B의 출력 랭크로 인덱스 게산
			for(j = 0;j < dotv->noutSuf; j++) sidx[dotv->outAxisSuf[j]] = tmp_idx[j];//B의 출력 인덱스 설정
		} else {//A의 조인 인덱스를 B의 출력 인덱스에 바로 설정.
			for(j = 0;j < dotv->noutSuf; j++) sidx[dotv->outAxisSuf[j]] = ridx[dotv->joAxisRet[j]];
		}
	} else if(dotv->bwGetOri == BWDIV_SUFF) {//A*B=C, B(ret)=A'*C,순전파때 B의 조인축은 역전파때 A의 출력축이되고 순전파때B의 출력축은
		doffset2idx(dotv->ndimRet, dotv->szRankRet, roff, ridx);//역전파때 C의 출력축이된다.여기서 A외C의 출력축을 정하고 밑에서 조인축
		if(dotv->noutPre > 1 || dotv->njoRet > 1) {				//을 따라 조인.
			soff = dsparse_idx2offset(dotv->njoRet, dotv->joRankRet, ridx, dotv->joAxisRet);//B의 조인랭크로 옵셋변환후
			doffset2idx(dotv->noutPre, dotv->outRankPre, soff, tmp_idx);//변환 옵셋과 A의 출력 랭크로 인덱스 게산
			for(j = 0;j < dotv->noutPre; j++) pidx[dotv->outAxisPre[j]] = tmp_idx[j];//A의 출력 인덱스 설정
		} else {//B의 조인 인덱스를 A의 출력 인덱스에 바로 설정.
			for(j = 0;j < dotv->noutPre; j++) pidx[dotv->outAxisPre[j]] = ridx[dotv->joAxisRet[j]];//전체 인덱스에 출력차원 인덱스 설정.
		}
		if(dotv->noutSuf > 1 || dotv->noutRet > 1) {
			soff = dsparse_idx2offset(dotv->noutRet, dotv->outRankRet, ridx, dotv->outAxisRet);//B의 출력랭크로 옵셋 변환후
			doffset2idx(dotv->noutSuf, dotv->outRankSuf, soff, tmp_idx);//변환 옵셋과 C의 출력 랭크로 인덱스 계산
			for(j = 0;j < dotv->noutSuf; j++) sidx[dotv->outAxisSuf[j]] = tmp_idx[j];//C의 출력 인덱스 설정
		} else {//B의 출력 인덱스를 C의 출력 인덱스에 바로 설정.
			for(j = 0;j < dotv->noutSuf; j++) sidx[dotv->outAxisSuf[j]] = ridx[dotv->outAxisRet[j]];
		}
	} else {
		poff = roff / sz_suf_shrink;//선형 공간상에서 두개 매트릭스가 곱해진 매트릭스상에서의 옵셋을 suffix매트릭스의
							//n_joint_axis dim을 제외한 사이즈로 나누어 prefix매트릭스상의 n_joint_axis dim이 제외된 옵셋을 획득한다.
		doffset2idx(dotv->noutPre, dotv->outRankPre, poff, tmp_idx);//최상위 축소차원이내 차원중에서 출력차원 인덱스 게산
		for(j = 0;j < dotv->noutPre; j++) pidx[dotv->outAxisPre[j]] = tmp_idx[j];//전체 인덱스에 출력차원 인덱스 설정.
		soff = roff % sz_suf_shrink;//suffix매트릭스상의 n_joint_axis dim이 제외된 옵셋을 계산
		doffset2idx(dotv->noutSuf, dotv->outRankSuf, soff, tmp_idx);
		for(j = 0;j < dotv->noutSuf; j++) sidx[dotv->outAxisSuf[j]] = tmp_idx[j];
	}
	//위에서 양측 획득된 매트릭스 인덱스에서 n_joint_axis차원들 인덱스만 순차적으로 증가하며 양측 원소 곱셈, 조인트 차원 인덱스 설정.
	for(i = 0, sum = 0;i < n_joint_axis; i++) {
		doffset2idx(dotv->njoPre, dotv->joRankPre, i, tmp_idx);
		for(j = 0;j < dotv->njoPre; j++) pidx[dotv->joAxisPre[j]] = tmp_idx[j];
		doffset2idx(dotv->njoSuf, dotv->joRankSuf, i, tmp_idx);
		for(j = 0;j < dotv->njoSuf; j++) sidx[dotv->joAxisSuf[j]] = tmp_idx[j];
		poff = didx2offset(pdim, prank, pidx);
		soff = didx2offset(sdim, srank, sidx);
		//printf("%f %f\n", *(m_pdot + poff), *(m_sdot + soff));
		sum += *(m_pdot + poff) * *(m_sdot + soff);
	}
	if(dotv->bwGetOri) {
		if(rplus != 1) *(m_rdot + roff) *= rplus;
	} else {
		if(rplus == 0) *(m_rdot + roff) = 0;
		else *(m_rdot + roff) *= rplus;
	}
	*(m_rdot + roff) += sum;
	//printf("%d %f\n", roff, *(m_rdot + roff));
}*/
/*
template<typename T>
__global__ void kdot_f2(void *pcxt, T *m_pdot, T *m_sdot, T *m_rdot, intt idx_origin, intt idx_width, T rplus, intt range, intt n)
{//그리드 사용 안는 버전
	DotVar *dotv = (DotVar *)pcxt;
	intt roff = idx_width * idx_origin + blockIdx.x * blockDim.x * range + threadIdx.x * range;
	//printf("%d %d %d\n", blockIdx.x, blockDim.x, threadIdx.x);
	if(roff >= n) return;
	if(n > roff + range) n = roff + range;

	intt ridx[MX_DIM], tmp_idx[MX_DIM], soff;
	intt rdim = dotv->ndimRet, *out_rank = dotv->outRank;
	intt *rrank = dotv->szRankRet;
	intt nout_pre = dotv->noutPre, nout = dotv->noutRank;
	intt njo_pre = dotv->njoPre, njo_suf = dotv->njoSuf;
	intt nout_suf = dotv->noutSuf, njo_ret = dotv->njoRet, *out_axis_ret = dotv->outAxisRet;
	intt *jo_axis_ret = dotv->joAxisRet;
	intt *out_rank_pre = dotv->outRankPre, *out_rank_suf = dotv->outRankSuf, *Jo_rank_ret = dotv->joRankRet;
	const bool bw_get_ori = dotv->bwGetOri, jdim_equal = dotv->jdimEqual, interv_out = dotv->intervOut, ret_first = dotv->retFirst;
	SparseRank *spr_pre_out = dotv->sprPreOut, *spr_pre_jo = dotv->sprPreJo, *spr_suf_out = dotv->sprSufOut, *spr_suf_jo = dotv->sprSufJo;
	intt po_idx[MX_DIM], pj_idx[MX_DIM], so_idx[MX_DIM], sj_idx[MX_DIM], i;
	T sum;
	//현 roff는 ret매트릭스가 기준인 옵셋이고 이것을 이번 매트릭스 곱의 양측 out axis rank기준으로 변환한다.
	_offset2idx2(nout_pre, nout, out_rank, roff, po_idx, so_idx);
	for(i = 0;i < nout_pre; i++) m_pdot += (po_idx[i] * spr_pre_out[i].rksz);
	for(i = 0;i < nout_suf; i++) m_sdot += (so_idx[i] * spr_suf_out[i].rksz);
	if(jdim_equal) {//양측 조인 랭크가 같으면 한번에 초기화
		for(i = 0;i < njo_pre; i++) pj_idx[i] = sj_idx[i] = 0;
	} else {
		for(i = 0;i < njo_pre; i++) pj_idx[i] = 0;
		for(i = 0;i < njo_suf; i++) sj_idx[i] = 0;
	}
	const intt nout_pre2 = nout_pre - 1, nout_suf2 = nout_suf - 1, njo_pre2 = njo_pre - 1, njo_suf2 = njo_suf - 1;
	const intt pjdim = spr_pre_jo[njo_pre2].rkdim, pjsz = spr_pre_jo[njo_pre2].rksz;
	const intt sjdim = spr_suf_jo[njo_suf2].rkdim, sjsz = spr_suf_jo[njo_suf2].rksz;
	const intt podim = spr_pre_out[nout_pre2].rkdim, posz = spr_pre_out[nout_pre2].rksz;
	const intt sodim = spr_suf_out[nout_suf2].rkdim, sosz = spr_suf_out[nout_suf2].rksz;
	intt i_pj = 0, i_sj = 0, i_po = po_idx[nout_pre2], i_so = so_idx[nout_suf2];
	for(;roff < n; roff++) {
		for(sum = 0;;) {//끝에서부터 계수, 조인 인덱스만 순차적으로 증가하며 양측 원소 곱셈, 조인트 차원 인덱스 설정
			//printf("%f %f\n", *(m_pdot + poff), *(m_sdot + soff));
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
		if(bw_get_ori) {//역전파, 역전파때는 양측의 출력 axis가 interval이 생기거나 순서가 역전되거나 조인축의 객수가 틀리면 
			if(interv_out) {//출력 매트릭스인 ret매트릭스와 out axis로 구성되는 랭크가 틀릴수있고 out axis rank기준 idx를
				//ret매트릭스 랭크를 사용하여 ret매트릭스 옵셋으로 변환한다. 옵셋변환되므로 리턴하여 한번에 device mem으로 복사한다.
				po_idx[nout_pre2] = i_po;//말단은 밑에서 인덱스 증가하지 않으므로 여기서 설정
				so_idx[nout_suf2] = i_so;
				if(ret_first) {//ret매트릭스가 dot의 첫번째에 위치하는 경우(순전파때 해당 매트릭스(pre or suf)의 first가 조인되지 않은 경우
					for(i = 0;i < nout_pre; i++) {//pref out idx를 ret매트릭스의 출력 axis idx에 설정
						ridx[out_axis_ret[i]] = po_idx[i];
					}
					if(njo_ret == 1 && nout_suf == 1) ridx[jo_axis_ret[0]] = so_idx[0];//변환없이 바로 설정
					else {//suf out idx가 ret매트릭스의 조인 axis가 된다. suf out idx를 suf out랭크를 사용하여 옵셋으로 변환뒤 
						_sparse_idx2offset(nout_suf, out_rank_suf, so_idx, soff);//이 옵셋을
						_offset2idx(njo_ret, Jo_rank_ret, soff, tmp_idx);//ret매트릭스의 조인 랭크를 사용하여 조인 axis위치에
						for(i = 0;i < njo_ret; i++) ridx[jo_axis_ret[i]] = tmp_idx[i];//인덱싱한다.
					}
				} else {
					for(i = 0;i < nout_suf; i++) {//suf out idx를 ret매트릭스의 출력 axis idx에 설정
						ridx[out_axis_ret[i]] = so_idx[i];
					}
					if(njo_ret == 1 && nout_pre == 1) ridx[jo_axis_ret[0]] = po_idx[0];//변환없이 바로 설정
					else {//pref out idx가 ret매트릭스의 조인 axis가 된다. pref out idx를 pref out랭크를 사용하여 옵셋으로 변환뒤 
						_sparse_idx2offset(nout_pre, out_rank_pre, po_idx, soff);//이 옵셋을
						_offset2idx(njo_ret, Jo_rank_ret, soff, tmp_idx);//ret매트릭스의 조인 랭크를 사용하여 조인 axis위치에
						for(i = 0;i < njo_ret; i++) ridx[jo_axis_ret[i]] = tmp_idx[i];//인덱싱한다.
					}
				}
				_idx2offset(rdim, rrank, ridx, soff);//최종 ret매트릭스에 인덱싱된 idx로 ret매트릭스 옵셋으로 변환한다.
				if(rplus != 1) *(m_rdot + soff) *= rplus;
				*(m_rdot + soff) += sum;
			} else {//옶셋 변환없이 바로 설정한다.
				if(rplus != 1) *(m_rdot + roff) *= rplus;
				*(m_rdot + roff) += sum;
			}
		} else {
			if(rplus == 0) *(m_rdot + roff) = 0;
			else *(m_rdot + roff) *= rplus;
			*(m_rdot + roff) += sum;
		}
		for(;;) {//출력 차원 인덱스의 하위 단위 suf out 차원 인덱스 증가
			if(sodim == ++i_so) {
				i_so = 0;//suf out 말단 차원 단위 올림
				i = nout_suf2;
O1:				m_sdot -= spr_suf_out[i].rktsz;
				if(--i < 0) break;//상위 단위 증가로
				else {
					if(spr_suf_out[i].rkdim == ++so_idx[i]) {
						so_idx[i] = 0;//suf out 중간 차원 단위 올림
						goto O1;
					} else {
						m_sdot += spr_suf_out[i].rksz;//다음 상위 단위(차원) 주소값 증가
						goto LA;//suf out 중간 차원 단위 증가
					}
				}
			} else {
				m_sdot += sosz;//말단 단위(차원) 주소값 증가
				goto LA;//suf out 말단 단위 증가
			}
		}
		for(;;) {//출력 차원 인덱스의 상위 단위 pre out 차원 인덱스 증가
			if(podim == ++i_po) {
				i_po = 0;
				i = nout_pre2;
O2:				m_pdot -= spr_pre_out[i].rktsz;
				if(--i < 0) goto LB2;
				else {
					if(spr_pre_out[i].rkdim == ++po_idx[i]) {
						po_idx[i] = 0;
						goto O2;
					} else {
						m_pdot += spr_pre_out[i].rksz;//다음 상위 단위(차원)값 증가
						break;//suf out 중간 차원 단위 증가
					}
				}
			} else {
				m_pdot += posz;
				break;
			}
		}
LA:;
	}
LB2:;
}
template<typename T>
intt gdot_t2(void *pcxt_dev, T *pdot_mdev, T *sdot_mdev, T *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, T rplus)
{
	dim3 block(WIDTH_BLOCK2(SMALL_BLOCK));//dot t2설명 참조
	intt range = idx_width / block.x;
	dim3 grid(idx_width % (block.x * range) ? 2 : 1);//idx_origin를 블럭 사이즈 단위로 인덱싱하므로 그리드는 1개이다.
	intt n = (rdot_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rdot_size);

	kdot_f2<T> << <grid, block >> > (pcxt_dev, pdot_mdev, sdot_mdev, rdot_mdev, idx_origin, idx_width, rplus, range, n);

	cudaDeviceSynchronize();
	return n - idx_origin * idx_width;
}*/
template<typename T>
__global__ void kdot_f2(void *pcxt, T *m_pdot, T *m_sdot, T *m_rdot, intt idx_origin, intt idx_width, T rplus, intt n)
{//그리스 사용 버전
	DotVar *dotv = (DotVar *)pcxt;
	intt roff = idx_width * idx_origin + blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d %d %d\n", blockIdx.x, blockDim.x, threadIdx.x);
	if(roff >= n) return;

	intt ridx[MX_DIM], tmp_idx[MX_DIM], soff;
	intt rdim = dotv->ndimRet, *out_rank = dotv->outRank;
	intt *rrank = dotv->szRankRet;
	intt nout_pre = dotv->noutPre, nout = dotv->noutRank;
	intt njo_pre = dotv->njoPre, njo_suf = dotv->njoSuf;
	intt nout_suf = dotv->noutSuf, njo_ret = dotv->njoRet, *out_axis_ret = dotv->outAxisRet;
	intt *jo_axis_ret = dotv->joAxisRet;
	intt *out_rank_pre = dotv->outRankPre, *out_rank_suf = dotv->outRankSuf, *Jo_rank_ret = dotv->joRankRet;
	const bool bw_get_ori = dotv->bwGetOri, jdim_equal = dotv->jdimEqual, interv_out = dotv->intervOut, ret_first = dotv->retFirst;
	SparseRank *spr_pre_out = dotv->sprPreOut, *spr_pre_jo = dotv->sprPreJo, *spr_suf_out = dotv->sprSufOut, *spr_suf_jo = dotv->sprSufJo;
	intt po_idx[MX_DIM], pj_idx[MX_DIM], so_idx[MX_DIM], sj_idx[MX_DIM], i;
	T sum;
	//현 roff는 ret매트릭스가 기준인 옵셋이고 이것을 이번 매트릭스 곱의 양측 out axis rank기준으로 변환한다.
	_offset2idx2(nout_pre, nout, out_rank, roff, po_idx, so_idx);
	for(i = 0;i < nout_pre; i++) m_pdot += (po_idx[i] * spr_pre_out[i].rksz);
	for(i = 0;i < nout_suf; i++) m_sdot += (so_idx[i] * spr_suf_out[i].rksz);
	if(jdim_equal) {//양측 조인 랭크가 같으면 한번에 초기화
		for(i = 0;i < njo_pre; i++) pj_idx[i] = sj_idx[i] = 0;
	} else {
		for(i = 0;i < njo_pre; i++) pj_idx[i] = 0;
		for(i = 0;i < njo_suf; i++) sj_idx[i] = 0;
	}
	const intt nout_pre2 = nout_pre - 1, nout_suf2 = nout_suf - 1, njo_pre2 = njo_pre - 1, njo_suf2 = njo_suf - 1;
	const intt pjdim = spr_pre_jo[njo_pre2].rkdim, pjsz = spr_pre_jo[njo_pre2].rksz;
	const intt sjdim = spr_suf_jo[njo_suf2].rkdim, sjsz = spr_suf_jo[njo_suf2].rksz;
	intt i_pj = 0, i_sj = 0, i_po = po_idx[nout_pre2], i_so = so_idx[nout_suf2];
	for(sum = 0;;) {//끝에서부터 계수, 조인 인덱스만 순차적으로 증가하며 양측 원소 곱셈, 조인트 차원 인덱스 설정
		//printf("%f %f\n", *(m_pdot + poff), *(m_sdot + soff));
		sum += *m_pdot * *m_sdot;
		if(jdim_equal) {//양측 조인 인덱스가 같으면 대표로 pre join인덱스로 제어
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
J2:;				m_sdot -= spr_suf_jo[i].rktsz;
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
	if(bw_get_ori) {//역전파, 역전파때는 양측의 출력 axis가 interval이 생기거나 순서가 역전되거나 조인축의 객수가 틀리면 
		if(interv_out) {//출력 매트릭스인 ret매트릭스와 out axis로 구성되는 랭크가 틀릴수있고 out axis rank기준 idx를
			//ret매트릭스 랭크를 사용하여 ret매트릭스 옵셋으로 변환한다. 옵셋변환되므로 리턴하여 한번에 device mem으로 복사한다.
			po_idx[nout_pre2] = i_po;//말단은 밑에서 인덱스 증가하지 않으므로 여기서 설정
			so_idx[nout_suf2] = i_so;
			if(ret_first) {//ret매트릭스가 dot의 첫번째에 위치하는 경우(순전파때 해당 매트릭스(pre or suf)의 first가 조인되지 않은 경우
				for(i = 0;i < nout_pre; i++) {//pref out idx를 ret매트릭스의 출력 axis idx에 설정
					ridx[out_axis_ret[i]] = po_idx[i];
				}
				if(njo_ret == 1 && nout_suf == 1) ridx[jo_axis_ret[0]] = so_idx[0];//변환없이 바로 설정
				else {//suf out idx가 ret매트릭스의 조인 axis가 된다. suf out idx를 suf out랭크를 사용하여 옵셋으로 변환뒤 
					_sparse_idx2offset(nout_suf, out_rank_suf, so_idx, soff);//이 옵셋을
					_offset2idx(njo_ret, Jo_rank_ret, soff, tmp_idx);//ret매트릭스의 조인 랭크를 사용하여 조인 axis위치에
					for(i = 0;i < njo_ret; i++) ridx[jo_axis_ret[i]] = tmp_idx[i];//인덱싱한다.
				}
			} else {
				for(i = 0;i < nout_suf; i++) {//suf out idx를 ret매트릭스의 출력 axis idx에 설정
					ridx[out_axis_ret[i]] = so_idx[i];
				}
				if(njo_ret == 1 && nout_pre == 1) ridx[jo_axis_ret[0]] = po_idx[0];//변환없이 바로 설정
				else {//pref out idx가 ret매트릭스의 조인 axis가 된다. pref out idx를 pref out랭크를 사용하여 옵셋으로 변환뒤 
					_sparse_idx2offset(nout_pre, out_rank_pre, po_idx, soff);//이 옵셋을
					_offset2idx(njo_ret, Jo_rank_ret, soff, tmp_idx);//ret매트릭스의 조인 랭크를 사용하여 조인 axis위치에
					for(i = 0;i < njo_ret; i++) ridx[jo_axis_ret[i]] = tmp_idx[i];//인덱싱한다.
				}
			}
			_idx2offset(rdim, rrank, ridx, soff);//최종 ret매트릭스에 인덱싱된 idx로 ret매트릭스 옵셋으로 변환한다.
			if(rplus) *(m_rdot + soff) += sum;
			else *(m_rdot + soff) = sum;
		} else {//옶셋 변환없이 바로 설정한다.
			if(rplus) *(m_rdot + roff) += sum;
			else *(m_rdot + roff) = sum;
		}
	} else {
		if(rplus) *(m_rdot + roff) += sum;
		else *(m_rdot + roff) = sum;
	}
}
template<typename T>
intt gdot_t2(void *pcxt_dev, T *pdot_mdev, T *sdot_mdev, T *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, T rplus)
{
	dim3 block(WIDTH_BLOCK2(SMALL_BLOCK));//dot t2설명 참조
	dim3 grid((idx_width + block.x - 1) / block.x);
	intt n = (rdot_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rdot_size);

	kdot_f2<T> << <grid, block >> > (pcxt_dev, pdot_mdev, sdot_mdev, rdot_mdev, idx_origin, idx_width, rplus, n);

	cudaDeviceSynchronize();
	return n - idx_origin * idx_width;
}
intt gdot_f2(void *pcxt_dev, floatt *pdot_mdev, floatt *sdot_mdev, floatt *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, floatt rplus)
{
	return gdot_t2<floatt>(pcxt_dev, pdot_mdev, sdot_mdev, rdot_mdev, rdot_size,
		idx_origin, idx_width, rplus);
}
intt gdot_f2(void *pcxt_dev, intt *pdot_mdev, intt *sdot_mdev, intt *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, intt rplus)
{
	return gdot_t2<intt>(pcxt_dev, pdot_mdev, sdot_mdev, rdot_mdev, rdot_size,
		idx_origin, idx_width, rplus);
}
