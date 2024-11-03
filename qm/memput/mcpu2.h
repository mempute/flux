#pragma once
#include "mcpu.h"

#define OFF_DOTM_TO_ORIGIN(roff, op, sz_suf_shrink, n_joint_axis, mrank, jo_top_axis, mdim, off_dot, off_origin, axis_fringe_sz, d, m, midx, mac_off2idx) {\
	off_origin = roff op sz_suf_shrink;/*���� �����󿡼� �ΰ� ��Ʈ������ ������ ��Ʈ�����󿡼��� �ɼ��� suffix��Ʈ������ n_joint_axis dim��*/\
								/*������ ������� ������ prefix��Ʈ�������� n_joint_axis dim�� ���ܵ� �ɼ��� ȹ���Ѵ�.*/\
	/*off_origin = off_dot;/* * n_joint_axis;/*��.�� ȹ��� �ɼ¿� ���ܵ� n_joint_axis dim�� ���Ͽ� ������ prefix��Ʈ�������� �ɼ� ȹ��*/\
	/*axis_fringe_sz = *(mrank + jo_top_axis);/*prefix��Ʈ������ �������(n_joint_axis) ���� ������*/\
	/*d = off_origin / axis_fringe_sz;\
	/*m = off_origin % axis_fringe_sz;\
	/*off_origin = d * axis_fringe_sz;/*������� �̻� �ɼ�, ������� �̻� �ɼ��� ������� ������ ������ �̵�*/\
	/*mac_off2idx(mdim, mrank, off_origin, midx);/*������� �̻� �ɼ� ��Ʈ���� �ε��� ���, ����.*/\
	/*off_origin = m / n_joint_axis;/*������� ���� �ɼ�. ������� ���ϴ� �Ѱ��� �̵��ϴµ� ��)���� ������� ���� ������ ���Ѱ���*/\
									/*�ٽ� ������ ������� �����߿��� ��� ���� ���ҵ��� �ɼ� ���.*/\
}
/*typedef struct {//dot version 1
	void *pheadVar;
	sytet joTopAxisPre, joTopAxisSuf;//joint axis�߿��� pre, suf ���� ���� ���� ����(0�� �������� ���� ����) �� �ε���
	sytet bwGetOri;
	bool intervOut;
	intt szShrinkSuf;//suff��Ʈ�������� nJointAxis�� �����ϰ� ������ dim���� ���� ��ҵ� ������
	intt ndimPre, ndimSuf, ndimRet;
	intt nJointAxis;//joint axis���� dim�� ���� ��
	intt njoPre, njoSuf, njoRet;//����Ʈ�Ǵ� ���� ����
	intt joAxisPre[MX_DIM], joAxisSuf[MX_DIM], joAxisRet[MX_DIM];//����Ʈ�Ǵ� ���� �ε���
	intt joRankPre[MX_DIM], joRankSuf[MX_DIM], joRankRet[MX_DIM];//�������� ���Һ��� �������� ���� ��ũ ������, dot ��� ��翡�� ������� �ȴ�.
	intt noutPre, noutSuf, noutRet;//��µǴ� ���� ����
	intt outAxisPre[MX_DIM], outAxisSuf[MX_DIM], outAxisRet[MX_DIM];//��µǴ� ���� �ε���
	intt outRankPre[MX_DIM], outRankSuf[MX_DIM], outRankRet[MX_DIM];//����Ʈ�� �ƴ� ���� ��ũ ������, dot ��� shape�� �ȴ�.
	intt szRankPre[MX_DIM], szRankSuf[MX_DIM], szRankRet[MX_DIM];//���� ��Ʈ���� ������ dot�Ǳ� ���� ������ ���� ��ũ ������
	intt shareUnit;//dot ���� �������� ȣȯ��.
} DotVar;
template<typename T>
intt cdot_t(void *pcxt, T *m_pdot, T *m_sdot, T *m_rdot, intt rdot_size, intt idx_origin, intt idx_width, floatt rplus)
{
	DotVar *dotv = (DotVar *)pcxt;
	intt roff = idx_origin * idx_width, toff;
	intt n = (rdot_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rdot_size);
	intt pidx[MX_DIM], sidx[MX_DIM], ridx[MX_DIM], tmp_idx[MX_DIM], poff, soff;
	intt sz_suf_shrink = dotv->szShrinkSuf, jo_top_axis_p = dotv->joTopAxisPre, jo_top_axis_s = dotv->joTopAxisSuf;
	intt pdim = dotv->ndimPre, sdim = dotv->ndimSuf, n_joint_axis = dotv->nJointAxis;
	intt *prank = dotv->szRankPre, *srank = dotv->szRankSuf, i, j;
	T sum;

	for(;roff < n; roff++) {//���������� �Ϲ� ��İ� ���������� ���� ���� ����
		if(dotv->bwGetOri == BWDIV_PREF) {//A*B=C, A(ret)=C*B',�����Ķ� A�� ������� �����Ķ� C�� ������̵ǰ� �����Ķ� A�� ��������
			offset2idx(dotv->ndimRet, dotv->szRankRet, roff, ridx);//�����Ķ� B�� ������̵ȴ�.���⼭ B��C�� ������� ���ϰ� �ؿ���
			if(dotv->noutPre > 1 || dotv->noutRet > 1) {			//�������� ���� ����
				soff = sparse_idx2offset(dotv->noutRet, dotv->outRankRet, ridx, dotv->outAxisRet);//A�� ��·�ũ�� �ɼº�ȯ��
				offset2idx(dotv->noutPre, dotv->outRankPre, soff, tmp_idx);//��ȯ �ɼ°� C�� ��� ��ũ�� �ε��� �Ի�
				for(j = 0;j < dotv->noutPre; j++) pidx[dotv->outAxisPre[j]] = tmp_idx[j];//C�� ��� �ε��� ����
			} else {//A�� ��� �ε����� C�� ��� �ε����� �ٷ� ����.
				for(j = 0;j < dotv->noutPre; j++) pidx[dotv->outAxisPre[j]] = ridx[dotv->outAxisRet[j]];//��ü �ε����� ������� �ε��� ����.
			}
			if(dotv->noutSuf > 1 || dotv->njoRet > 1) {
				soff = sparse_idx2offset(dotv->njoRet, dotv->joRankRet, ridx, dotv->joAxisRet);//A�� ���η�ũ�� �ɼº�ȯ��
				offset2idx(dotv->noutSuf, dotv->outRankSuf, soff, tmp_idx);//��ȯ �ɼ°� B�� ��� ��ũ�� �ε��� �Ի�
				for(j = 0;j < dotv->noutSuf; j++) sidx[dotv->outAxisSuf[j]] = tmp_idx[j];//B�� ��� �ε��� ����
			} else {//A�� ���� �ε����� B�� ��� �ε����� �ٷ� ����.
				for(j = 0;j < dotv->noutSuf; j++) sidx[dotv->outAxisSuf[j]] = ridx[dotv->joAxisRet[j]];
			}
		} else if(dotv->bwGetOri == BWDIV_SUFF) {//A*B=C, B(ret)=A'*C,�����Ķ� B�� �������� �����Ķ� A�� ������̵ǰ� �����Ķ�B�� �������
			offset2idx(dotv->ndimRet, dotv->szRankRet, roff, ridx);//�����Ķ� C�� ������̵ȴ�.���⼭ A��C�� ������� ���ϰ� �ؿ��� ������
			if(dotv->noutPre > 1 || dotv->njoRet > 1) {				//�� ���� ����.
				soff = sparse_idx2offset(dotv->njoRet, dotv->joRankRet, ridx, dotv->joAxisRet);//B�� ���η�ũ�� �ɼº�ȯ��
				offset2idx(dotv->noutPre, dotv->outRankPre, soff, tmp_idx);//��ȯ �ɼ°� A�� ��� ��ũ�� �ε��� �Ի�
				for(j = 0;j < dotv->noutPre; j++) pidx[dotv->outAxisPre[j]] = tmp_idx[j];//A�� ��� �ε��� ����
			} else {//B�� ���� �ε����� A�� ��� �ε����� �ٷ� ����.
				for(j = 0;j < dotv->noutPre; j++) pidx[dotv->outAxisPre[j]] = ridx[dotv->joAxisRet[j]];//��ü �ε����� ������� �ε��� ����.
			}
			if(dotv->noutSuf > 1 || dotv->noutRet > 1) {
				soff = sparse_idx2offset(dotv->noutRet, dotv->outRankRet, ridx, dotv->outAxisRet);//B�� ��·�ũ�� �ɼ� ��ȯ��
				offset2idx(dotv->noutSuf, dotv->outRankSuf, soff, tmp_idx);//��ȯ �ɼ°� C�� ��� ��ũ�� �ε��� ���
				for(j = 0;j < dotv->noutSuf; j++) sidx[dotv->outAxisSuf[j]] = tmp_idx[j];//C�� ��� �ε��� ����
			} else {//B�� ��� �ε����� C�� ��� �ε����� �ٷ� ����.
				for(j = 0;j < dotv->noutSuf; j++) sidx[dotv->outAxisSuf[j]] = ridx[dotv->outAxisRet[j]];
			}
		} else {
			poff = roff / sz_suf_shrink;//���� �����󿡼� �ΰ� ��Ʈ������ ������ ��Ʈ�����󿡼��� �ɼ��� suffix��Ʈ������
								//n_joint_axis dim�� ������ ������� ������ prefix��Ʈ�������� n_joint_axis dim�� ���ܵ� �ɼ��� ȹ���Ѵ�.
			offset2idx(dotv->noutPre, dotv->outRankPre, poff, tmp_idx);//�ֻ��� ��������̳� �����߿��� ������� �ε��� �Ի�
			for(j = 0;j < dotv->noutPre; j++) pidx[dotv->outAxisPre[j]] = tmp_idx[j];//��ü �ε����� ������� �ε��� ����.
			soff = roff % sz_suf_shrink;//suffix��Ʈ�������� n_joint_axis dim�� ���ܵ� �ɼ��� ���
			offset2idx(dotv->noutSuf, dotv->outRankSuf, soff, tmp_idx);
			for(j = 0;j < dotv->noutSuf; j++) sidx[dotv->outAxisSuf[j]] = tmp_idx[j];
		}
		//������ ���� ȹ��� ��Ʈ���� �ε������� n_joint_axis������ �ε����� ���������� �����ϸ� ���� ���� ����, ����Ʈ ���� �ε��� ����.
		for(i = 0, sum = 0;i < n_joint_axis; i++) {
			offset2idx(dotv->njoPre, dotv->joRankPre, i, tmp_idx);
			for(j = 0;j < dotv->njoPre; j++) pidx[dotv->joAxisPre[j]] = tmp_idx[j];
			offset2idx(dotv->njoSuf, dotv->joRankSuf, i, tmp_idx);
			for(j = 0;j < dotv->njoSuf; j++) sidx[dotv->joAxisSuf[j]] = tmp_idx[j];
			poff = idx2offset(pdim, prank, pidx);
			soff = idx2offset(sdim, srank, sidx);
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
	}
	return n - idx_origin * idx_width;
}*/

template<typename T>
intt cdot_t2(void *pcxt, T *m_pdot, T *m_sdot, T *m_rdot, intt rdot_size, intt idx_origin, intt idx_width, floatt rplus)
{
	DotVar *dotv = (DotVar *)pcxt;
	intt roff = idx_origin * idx_width;
	const intt n = (rdot_size > (idx_origin + 1) * idx_width ? (idx_origin + 1) * idx_width : rdot_size);
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
	//�� roff�� ret��Ʈ������ ������ �ɼ��̰� �̰��� �̹� ��Ʈ���� ���� ���� out axis rank�������� ��ȯ�Ѵ�.
	_offset2idx2(nout_pre, nout, out_rank, roff, po_idx, so_idx);
	for(i = 0;i < nout_pre; i++) m_pdot += (po_idx[i] * spr_pre_out[i].rksz);
	for(i = 0;i < nout_suf; i++) m_sdot += (so_idx[i] * spr_suf_out[i].rksz);
	if(jdim_equal) {//���� ���� ��ũ�� ������ �ѹ��� �ʱ�ȭ
		for(i = 0;i < njo_pre; i++) pj_idx[i] = sj_idx[i] = 0;
	} else {
		for(i = 0;i < njo_pre; i++) pj_idx[i] = 0;
		for(i = 0;i < njo_suf; i++) sj_idx[i] = 0;
	}
	/*intt j, k, M = spr_pre_out[0].rkdim, lda = spr_pre_jo[0].rkdim, N = spr_suf_out[0].rkdim, ldb = spr_suf_jo[0].rkdim, K = lda;
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
	const intt nout_pre2 = nout_pre - 1, nout_suf2 = nout_suf - 1, njo_pre2 = njo_pre - 1, njo_suf2 = njo_suf - 1;
	const intt pjdim = spr_pre_jo[njo_pre2].rkdim, pjsz = spr_pre_jo[njo_pre2].rksz;
	const intt sjdim = spr_suf_jo[njo_suf2].rkdim, sjsz = spr_suf_jo[njo_suf2].rksz;
	const intt podim = spr_pre_out[nout_pre2].rkdim, posz = spr_pre_out[nout_pre2].rksz;
	const intt sodim = spr_suf_out[nout_suf2].rkdim, sosz = spr_suf_out[nout_suf2].rksz;
	intt i_pj = 0, i_sj = 0, i_po = po_idx[nout_pre2], i_so = so_idx[nout_suf2];
	for(;roff < n; roff++) {
		for(sum = 0;;) {//���������� ���, ���� �ε����� ���������� �����ϸ� ���� ���� ����, ����Ʈ ���� �ε��� ����
			//printf("%f %f\n", *(m_pdot + poff), *(m_sdot + soff));
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
		if(bw_get_ori) {//������, �����Ķ��� ������ ��� axis�� interval�� ����ų� ������ �����ǰų� �������� ������ Ʋ���� 
			if(interv_out) {//��� ��Ʈ������ ret��Ʈ������ out axis�� �����Ǵ� ��ũ�� Ʋ�����ְ� out axis rank���� idx��
				//ret��Ʈ���� ��ũ�� ����Ͽ� ret��Ʈ���� �ɼ����� ��ȯ�Ѵ�. �ɼº�ȯ�ǹǷ� �����Ͽ� �ѹ��� device mem���� �����Ѵ�.
				po_idx[nout_pre2] = i_po;//������ �ؿ��� �ε��� �������� �����Ƿ� ���⼭ ����
				so_idx[nout_suf2] = i_so;
				if(ret_first) {//ret��Ʈ������ dot�� ù��°�� ��ġ�ϴ� ���(�����Ķ� �ش� ��Ʈ����(pre or suf)�� first�� ���ε��� ���� ���
					for(i = 0;i < nout_pre; i++) {//pref out idx�� ret��Ʈ������ ��� axis idx�� ����
						ridx[out_axis_ret[i]] = po_idx[i];
					}
					if(njo_ret == 1 && nout_suf == 1) ridx[jo_axis_ret[0]] = so_idx[0];//��ȯ���� �ٷ� ����
					else {//suf out idx�� ret��Ʈ������ ���� axis�� �ȴ�. suf out idx�� suf out��ũ�� ����Ͽ� �ɼ����� ��ȯ�� 
						_sparse_idx2offset(nout_suf, out_rank_suf, so_idx, soff);//�� �ɼ���
						_offset2idx(njo_ret, Jo_rank_ret, soff, tmp_idx);//ret��Ʈ������ ���� ��ũ�� ����Ͽ� ���� axis��ġ��
						for(i = 0;i < njo_ret; i++) ridx[jo_axis_ret[i]] = tmp_idx[i];//�ε����Ѵ�.
					}
				} else {
					for(i = 0;i < nout_suf; i++) {//suf out idx�� ret��Ʈ������ ��� axis idx�� ����
						ridx[out_axis_ret[i]] = so_idx[i];
					}
					if(njo_ret == 1 && nout_pre == 1) ridx[jo_axis_ret[0]] = po_idx[0];//��ȯ���� �ٷ� ����
					else {//pref out idx�� ret��Ʈ������ ���� axis�� �ȴ�. pref out idx�� pref out��ũ�� ����Ͽ� �ɼ����� ��ȯ�� 
						_sparse_idx2offset(nout_pre, out_rank_pre, po_idx, soff);//�� �ɼ���
						_offset2idx(njo_ret, Jo_rank_ret, soff, tmp_idx);//ret��Ʈ������ ���� ��ũ�� ����Ͽ� ���� axis��ġ��
						for(i = 0;i < njo_ret; i++) ridx[jo_axis_ret[i]] = tmp_idx[i];//�ε����Ѵ�.
					}
				}
				_idx2offset(rdim, rrank, ridx, soff);//���� ret��Ʈ������ �ε��̵� idx�� ret��Ʈ���� �ɼ����� ��ȯ�Ѵ�.
				if(rplus != 1) *(m_rdot + soff) *= rplus;
				*(m_rdot + soff) += sum;
			} else {//���� ��ȯ���� �ٷ� �����Ѵ�.
				if(rplus != 1) *(m_rdot + roff) *= rplus;
				*(m_rdot + roff) += sum;
			}
		} else {
			if(rplus == 0) *(m_rdot + roff) = 0;
			else *(m_rdot + roff) *= rplus;
			*(m_rdot + roff) += sum;
		}
		for(;;) {//��� ���� �ε����� ���� ���� suf out ���� �ε��� ����
			if(sodim == ++i_so) {
				i_so = 0;//suf out ���� ���� ���� �ø�
				i = nout_suf2;
O1:				m_sdot -= spr_suf_out[i].rktsz;
				if(--i < 0) break;//���� ���� ������
				else {
					if(spr_suf_out[i].rkdim == ++so_idx[i]) {
						so_idx[i] = 0;//suf out �߰� ���� ���� �ø�
						goto O1;
					} else {
						m_sdot += spr_suf_out[i].rksz;//���� ���� ����(����) �ּҰ� ����
						goto LA;//suf out �߰� ���� ���� ����
					}
				}
			} else {
				m_sdot += sosz;//���� ����(����) �ּҰ� ����
				goto LA;//suf out ���� ���� ����
			}
		}
		for(;;) {//��� ���� �ε����� ���� ���� pre out ���� �ε��� ����
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
						m_pdot += spr_pre_out[i].rksz;//���� ���� ����(����)�� ����
						break;//suf out �߰� ���� ���� ����
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
	return n - idx_origin * idx_width;
}