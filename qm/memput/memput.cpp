

#include <iostream>
#include <list>
#ifdef OPT_WIN
#include <windows.h>
#else
#include <unistd.h>
#include <stdarg.h>
#endif
#include "intracore.h"
#include "intracore2.h"
#include "baper.h"

void *Typer::operator new(size_t size, Tracer *tcr)
{
	return TRACER(tcr)->xalloc(size);
}
void Flux::init(Tracer *tcr, ubytet qtype, ubytet fxtype, vinitfp vfp, const bytet *name)
{
	qType = qtype;
	fxType = fxtype;
	quantum = nullptr;
	nRefer = 0;
	bwAp = bwLink = directLink = nullptr;
	fwAp = nullptr;
	fxTcr = tcr;
	backwv = 0;
	ibRefer = 0;
	termifx = false;
	//nJoint = 1;
	vinitter = vfp;
	if(name || fxtype == memput::mp::trainable) {
		fxName = (bytet *)TRACER(tcr)->namewrite(name);
		TRACER(tcr)->listw(this, false);
		trainherit = 1;
	} else {
		fxName = nullptr;
		TRACER(tcr)->listw(this, true);
		trainherit = 0;
	}
	meanAfter = 0;
	bwbreak = 0;
	partitionfx = 0;
	ofxArrange = 0;
	changeGround = false;
	if(fxtype == variable || fxtype == persistant) {//variable, persistant�� ������ �ƿ��� �� ��ġ
		scaleout = fshape[0];//������ �����Ǿ� �ܺο��� ��ġ Ȯ��ɶ� ���� �����Ͼƿ��� ������ ���ǹǷ�
							//���ο��� �ٸ� �÷����� ��ġ �����Ǵ� �������� ������� �ʴ´�.
	} else scaleout = 0;//apply�� ����������� �ҽ��κ��� ��ӵǾ� �����ǰ�, trainable�� ��õ���μ� Ȯ��Ұ��� �����Ѵ�.
	//ds = gs = nullptr;
}
intt Flux::sizef(void) //�÷��� ���� ����
{
	intt sz;
	SIZE_SHAPE(fdim, fshape, sz);
	return sz;
}
intt Flux::sizefx(bool t_size) //t_size - true�̸� Ÿ�� ������, false�̸� �÷��� ����Ʈ ������
{
	return TENSOR(quantum)->mxData->sizem(t_size);
}
#define REDUCE_SCALE(vscale) {\
	intt div = fshape[0] / scaleout;\
	if(fshape[0] % scaleout || div < 1) throwFault(-1, "scale v error %d %d %d\n", fshape[0], scaleout, div);\
	vscale = axid[0] / div;\
	if(axid[0] % div || vscale < 1) throwFault(-1, "scale v error2  %d %d\n", axid[0], vscale);\
}
//��ġȮ�� �������� ù��° ���� ������ ��� - 0������ ��ġ�̿��� ������ ���Ե������� 0�� ���������� ��ġ�� 
//������ ���� �Ѽ��� �̹� ����Ǵ� ��ġ ������ ���Ѱ��� 0���� ����� �ȴ�.
#define FIRST_SHAPE_SCALE(fx, bsz_expand) ((fx->fshape[0] / fx->scaleout) * bsz_expand)
intt Flux::sizefx2(intt n, bool bsize) //n- batch size, bsize - true�̸� byte ���� size false�̸� ����
{
	if(n > 0 && scaleout > 0) {//variable�÷����κ��� ��������� ����� �÷����϶��� ��ġ n���� �������� ������(�׷���� ������ ��Ʈ������ ����)�� ����Ѵ�
		return TENSOR(quantum)->mxData->sizem2(FIRST_SHAPE_SCALE(this, n), bsize) * 2;// * 2 == data + grad
	} else {//�� ��ġ ���¿����� �׷���� ������ ��Ʈ������ ���� ������
		return (bsize ? TENSOR(quantum)->mxData->sizem() : fxSize) * 2;// * 2 == data + grad
	}
}
void Flux::reentrance(bool on)
{
	bwbreak = on;//vsync���� �� �÷����� ���������� �����ǰ� �Ͽ� ������ ���࿡�� �� �÷��� ���ĺ��� ����ǰ� �Ѵ�.
				//�Ʒ� �������� ����� ���൵ ���⼭ ���߰� �ҷ��� �Ʒ� �Լ��� ����ؾ� �Ѵ�.->tcfnr�� ���� ����� ���� �ȵ�.
}
void Flux::switchTrace(Tracer *tcr, bool on)
{
	bwbreak = on;//���� ����, ����� ������ �Ʒ� ���ܿ��� tcr�� �ٸ��� �����ϹǷ� ���⼭ �����.->tcfnr.�� ���� ���������� ����� ���� �ȵ�.
	fxTcr = tcr;//�� �÷��� ���� �����Ǵ� apply flux���� �־��� tcr�� �����Ѱ� �Ѵ�.
}
#define TRCB(tcr) TRACER(tcr)->hbaper
Flux *Flux::switchout(Flux *other)
{
	TRCB(fxTcr)->bswitch_out(this, other);
	Flux *fxo = new(fxTcr)Flux(fxTcr, fshape, fdim, qType, apply, nullx, nullx);
	ApSwitchout *apo = new(fxTcr)ApSwitchout(TRACER(fxTcr), this, other, fxo);

	backend_ps(this, other, fxo, apo);
	return fxo;
}
Flux::Flux(Tracer *tcr, intt *axid, intt ndim, ubytet qtype, ubytet fxtype, vinitfp vfp, const bytet *name)
{
	if(ndim >= MX_DIM) throwFault(-1, "flex out of range\n");
	
	fdim = ndim;
	memcpy(fshape, axid, ndim * sizeof(intt));
	init(tcr, qtype, fxtype, vfp, name);
	fxSize = sizef();
	instTens(false);//fnisqgt.
	TRCB(tcr)->bflux(this, axid, ndim, qtype, fxtype, vfp, name);
}
Flux::Flux(Tracer *tcr, initializer_list<intt> axid, ubytet qtype, ubytet fxtype, vinitfp vfp, const bytet *name)
{
	intt i = 0;
	
	fdim = (intt)axid.size();
	if(fdim >= MX_DIM) throwFault(-1, "flex out of range\n");
	for(auto iter = axid.begin();i < fdim; iter++, i++) {
		if(*iter == 0) throwFault(-1, "dimension not allowed\n");
		fshape[i] = (*iter < 0 ? 1 : *iter);
	}
	init(tcr, qtype, fxtype, vfp, name);
	fxSize = sizef();
	instTens(false);//fnisqgt.
	TRCB(tcr)->bflux(this, fshape, fdim, qtype, fxtype, vfp, name);
}
Flux::Flux(Tracer *tcr, vector<intt> &axid, ubytet qtype, ubytet fxtype, vinitfp vfp, const bytet *name, Flux *mast)
{
	intt i = 0;
	
	fdim = (intt)axid.size();
	if(fdim >= MX_DIM) throwFault(-1, "flex out of range\n");
	for(auto iter = axid.begin();i < fdim; iter++, i++) {
		if(*iter == 0) throwFault(-1, "dimension not allowed\n");
		fshape[i] = (*iter < 0 ? 1 : *iter);
	}
	init(tcr, qtype, fxtype, vfp, name);
	fxSize = sizef();
	instTens(false, -1, mast);//fnisqgt.
	TRCB(tcr)->bflux(this, fshape, fdim, qtype, fxtype, vfp, name);
}
void Flux::instTens(bool inst, intt gid, Flux *mast)
{
	try {
		//�� �÷����� reshape���� ��� �����Ǵ� ���̸� ������ reshape�Ǵ� �÷����� �����Ѵ�.
		ptrMastfx = (mast ? (mast->ptrMastfx ? mast->ptrMastfx : mast) : nullptr);
		if(quantum == nullptr) {
			quantum = new(fxTcr)Tensor(fxTcr, qType);
			inst = true;
		}
		if(inst) {
			//if(fxSize < 0) throwFault(-1, "first dimension not defined\n");
			((Tensor *)quantum)->instTensor(fdim, fshape, gid, ptrMastfx);
			fxSize = sizef();
		}
	} catch(FaultObj fo) {
		throwFault(-1, "inst tens\n%s", fo.fltmsg);
	}
}
void Flux::fdims(intt ndim, intt *pdim)
{
	intt i = 0;
	bool inst = false;

	if(ndim == fdim) {
		for(;i < ndim && *(pdim + i) == fshape[i]; i++);
		if(ndim != fdim || i < ndim) inst = true;
	} else inst = true;
	
	if(inst) {
		fdim = ndim;
		memcpy(fshape, pdim, ndim * sizeof(intt));
	}
	instTens(inst);
}
void Flux::fdims(initializer_list<intt> axid)
{
	intt pdim[MX_DIM], ndim, i = 0;

	ndim = (intt)axid.size();
	if(ndim >= MX_DIM) throwFault(-1, "flex out of range\n");
	for(auto iter = axid.begin();i < fdim; iter++, i++) pdim[i] = *iter;

	fdims(ndim, pdim);
}
bool Flux::checkInBwfx(bool lock)
{
	if(TRACER(fxTcr)->bwVersion != backwv) throwFault(-1, "check in bw version fault\n");
	if(lock) LOCK_MUT_(TRACER(fxTcr)->mutTrc);
	//if(++TRACER(fxTcr)->ibwFork > TRACER(fxTcr)->nbwFork) {
	//	//exit(1);//mmm
	//}
	if(++ibRefer == nbRefer) {
		if(lock) UNLOCK_MUT_(TRACER(fxTcr)->mutTrc);
		//printf("check flux bw ap: %d num ref: %d ref cnt: %d pass\n", ((Apply *)bwAp)->apCode, nbRefer, ibRefer);
		return true;
	} else {
		//printf("check flux bw ap: %d num ref: %d ref cnt: %d skip\n", ((Apply *)bwAp)->apCode, nbRefer, ibRefer);
		if(ibRefer > nbRefer) {
			if(lock) UNLOCK_MUT_(TRACER(fxTcr)->mutTrc);
			throwFault(-1, "check in fault\n");
		} else {
			if(lock) UNLOCK_MUT_(TRACER(fxTcr)->mutTrc);
			return false;
		}
	}
}
bool Flux::checkInBwfx2(void)
{
	LOCK_MUT_(TRACER(fxTcr)->mutTrc);
	if(checkInBwfx(0) && bwAp && ((Apply *)bwAp)->checkInBwap()) {
		UNLOCK_MUT_(TRACER(fxTcr)->mutTrc);
		return true;
	}
	UNLOCK_MUT_(TRACER(fxTcr)->mutTrc);
	return false;
}
void Flux::referenced(Typer *ap)
{
	if(TRACER(fxTcr)->directExec) return;
	if(termifx) throwFault(-1, "can not refer\n");
	Contact *fxpc = new(fxTcr)Contact;

	APPEND_LIST(fwAp, fxpc);
	fxpc->vcontact = ap;
	nRefer++;//��� ���� �׽�Ʈ �����μ� �׷��� ������� ��� ����.
	if(trainherit) ((Apply *)ap)->trainherit = 1;
}
void Flux::projected(Typer *ap)
{
	FxAnchor *fxa = (FxAnchor *)TRACER(fxTcr)->xalloc(sizeof(FxAnchor));
	fxa->fxPoint = this;
	APPEND_LIST(((Apply *)ap)->lapOuput, fxa);

	if(TRACER(fxTcr)->directExec) directLink = ap;
	else {
		bwLink = ap;
		if(((Apply *)ap)->trainherit) {
			bwAp = ap;
			trainherit = 1;
		}//else �׷��� �������� �ռ� �÷����߿� Ʈ���ξ���� ������ �������� �ʿ�����Ƿ� ����� ü�� �������� �ʴ´�.
	}
	
	//bwAp = ap;//�׶��� ����ʰ� ������ ���� �׽�Ʈ �Ҷ� �� ������ �����Ѵ�.
}
void Flux::exec(void *tcxt)
{
	Matrixr *mx = nullx;
	if(directLink == nullx) throwFault(-1, "can not direct exec\n");
	((Apply *)directLink)->forward(tcxt ? (TContext *)tcxt : TRACER(fxTcr)->trcCxt(((Apply *)directLink)->oidground()), mx);
}
void Flux::backward(void) //��� ���� �׽�Ʈ ��
{
	Matrixr *mx = nullx;
	if((nRefer > 0 ? --nRefer : nRefer)) return;

	((Apply *)directLink)->backward(TRACER(fxTcr)->trcCxt(((Apply *)directLink)->oidground()), mx);
}
void Flux::backend_nstep(Flux *fxp, Flux *fxs, Flux *fxo, void *ap)
{
	if(TRACER(fxTcr)->prompt || ((Apply *)ap)->loadOnExec) {//���߿� �����Ͽ� loadOnExec���� ����
		Matrixr *mx = nullx;
		if(fxp && (fxp->quantum == nullptr || TENSOR(fxp->quantum)->mxData->begin_p() == nullptr)) throwFault(-1, "not intantiate\n");
		if(fxs && (fxs->quantum == nullptr || TENSOR(fxs->quantum)->mxData->begin_p() == nullptr)) throwFault(-1, "not intantiate\n");
		//fnisqgt.if(fxo) fxo->instTens(false);
		((Apply *)ap)->forward(TRACER(fxTcr)->trcCxt(((Apply *)ap)->oidground()), mx);
	}
}
void Flux::backend_ps(Flux *fxp, Flux *fxs, Flux *fxo, void *ap, intt vscale)
{
	if(fxo) {
		if(((Apply *)ap)->meanApply) fxo->meanAfter = 1;
		else if(fxp) {
			if(fxs) {
				if(fxp->meanAfter && fxs->meanAfter) fxo->meanAfter = 1;
			} else if(fxp->meanAfter) fxo->meanAfter = 1;
		} else if(fxs && fxs->meanAfter) fxo->meanAfter = 1;

		if(vscale) fxo->scaleout = vscale;
		if(fxp) {
			fxp->referenced((Apply *)ap);
			if(vscale == 0 && fxp->scaleout > 0) fxo->scaleout = fxp->scaleout;
		}
		if(fxs) {
			fxs->referenced((Apply *)ap);
			if(vscale == 0 && fxs->scaleout > 0) fxo->scaleout = fxs->scaleout;
		}
		if(fxo->fshape[0] < fxo->scaleout) {
			throwFault(-1, "backend scale out error %d %d\n", fxo->fshape[0], fxo->scaleout);
		}
		fxo->projected((Apply *)ap);
	} else {
		if(fxp) fxp->referenced((Apply *)ap);
		if(fxs) fxs->referenced((Apply *)ap);
	}
	backend_nstep(this, fxs, fxo, ap);
}
intt bro_check(intt cdim, intt cxid[], intt ndim, intt axid[])
{
	intt bro = 0;
	intt i, beg = 0;
	bool bro_flag, changed;

	if(cdim > ndim) beg = cdim - ndim;

	bro_flag = (axid[0] == 1 ? 1 : 0);
	for(i = 0;i < ndim; i++) {
		changed = 0;
		if(axid[i] == 1) {
			if(cxid[beg + i] == 1) continue;
			if(bro_flag == 0) {
				changed = 1;
				bro_flag = 1;
			}
			if(bro == 0) bro = -1;
		} else {
			if(bro_flag == 1) {
				changed = 1;
				bro_flag = 0;
			}
			if(bro == 0) bro = 1;
		}
		if(changed) {
			if(bro > 0) bro++;
			else bro--;
		}
	}
	return bro;
}
Flux *Flux::arithmetic(Flux *fxs, ubytet qtype, void *sval, sytet arith_op)
{//this(pref)�� ��Į��(sval)�� ��� fxs�� this�� �־� ȣ���Ѵ�.
	intt axid[MX_DIM];
	intt i = 0, j = 0, ndim;
	bool bro_one = 0, const_arith = 0, new_out = 1;
	TRCB(fxTcr)->barith(this, fxs, qtype, sval, arith_op);

	if(fxs && this != fxs) {
		if(fdim > fxs->fdim) {
			ndim = fdim;
			for(;i < fdim - fxs->fdim; i++) axid[i] = fshape[i];
			for(;i < fdim; i++, j++) {
				if(fshape[i] == 1) {
					axid[i] = fxs->fshape[j];
					//bro_one = 1;
				} else if(fxs->fshape[j] == 1) {
					axid[i] = fshape[i];
					//bro_one = 1;
				} else if(fshape[i] != fxs->fshape[j]) throwFault(-1, "dimension inconsistency\n");
				else axid[i] = fshape[i];
			}
		} else {
			ndim = fxs->fdim;
			for(;i < fxs->fdim - fdim; i++) axid[i] = fxs->fshape[i];
			for(;i < fxs->fdim; i++, j++) {
				if(fshape[j] == 1) {
					axid[i] = fxs->fshape[i];
					//bro_one = 1;
				} else if(fxs->fshape[i] == 1) {
					axid[i] = fshape[j];
					//bro_one = 1;
				} else if(fshape[j] != fxs->fshape[i]) throwFault(-1, "dimension inconsistency\n");
				else axid[i] = fshape[j];
			}
		}
		i = bro_check(ndim, axid, fdim, fshape);
		if(i > 2 || i < -3) bro_one = 1;
		else {
			i = bro_check(ndim, axid, fxs->fdim, fxs->fshape);
			if(i > 2 || i < -3) bro_one = 1;
		}
		if(fxType > apply && fxs->fxType > apply) const_arith = 1;
	} else {
		ndim = fdim;
		memcpy(axid, fshape, fdim * sizeof(intt));
		if(fxType > apply) {
			const_arith = 1;
			new_out = 0;
		}
	}
	Flux *fxo;
	if(new_out) fxo = new(fxTcr)Flux(fxTcr, axid, ndim, qType, apply);
	else fxo = this;
	/*if(const_arith) {//������ ���� �׶��� ������ �ƴ� ��� ���� ���̽�//setArithv();�� ȣ���ؾ� �ϹǷ� �Ұ�
		bytet svbuf[sizeof(unit)];
		if(qType != qtype) {
			adj_val_type(svbuf, sval, qType, qtype, 0);
			sval = svbuf;
			qtype = qType;
		}
		TENSOR(quantum)->mxData->marith(TRACER(fxTcr)->trcCxt(), (sval ? nullx : TENSOR(fxs->quantum)->mxData),
			TENSOR(fxo->quantum)->mxData, nullx, qtype, sval, nullx, arith_op);
	} else {*/
		ApArith *ap_arith = new(fxTcr)ApArith(TRACER(fxTcr), this, fxs, fxo, qtype, sval, arith_op, bro_one);
		backend_ps(this, fxs, fxo, ap_arith);
	//}
		//printf("222 %p %p %p %p %p\n", this, fxs, ap_arith, fwAp ? (Apply *)fwAp->vcontact : nullx, fxs && fxs->fwAp ? (Apply *)fxs->fwAp->vcontact : nullx);
	return fxo;
}
Flux *Flux::mul(Flux *fxs, longt sval)
{
	return arithmetic(fxs, tlong, &sval, AOP_MUL);
}
Flux *Flux::plus(Flux *fxs, longt sval)
{
	return arithmetic(fxs, tlong, &sval, AOP_PLUS);
}
Flux *Flux::div(Flux *fxs, longt sval)
{
	return arithmetic(fxs, tlong, &sval, AOP_DIV);
}
Flux *Flux::minus(Flux *fxs, longt sval)
{
	return arithmetic(fxs, tlong, &sval, AOP_MINUS);
}
Flux *Flux::mul(Flux *fxs, intt sval)
{
	return arithmetic(fxs, tint, &sval, AOP_MUL);
}
Flux *Flux::plus(Flux *fxs, intt sval)
{
	return arithmetic(fxs, tint, &sval, AOP_PLUS);
}
Flux *Flux::div(Flux *fxs, intt sval)
{
	return arithmetic(fxs, tint, &sval, AOP_DIV);
}
Flux *Flux::minus(Flux *fxs, intt sval)
{
	return arithmetic(fxs, tint, &sval, AOP_MINUS);
}
Flux *Flux::mul(Flux *fxs, doublet sval)
{
	return arithmetic(fxs, tdouble, &sval, AOP_MUL);
}
Flux *Flux::plus(Flux *fxs, doublet sval)
{
	return arithmetic(fxs, tdouble, &sval, AOP_PLUS);
}
Flux *Flux::div(Flux *fxs, doublet sval)
{
	return arithmetic(fxs, tdouble, &sval, AOP_DIV);
}
Flux *Flux::minus(Flux *fxs, doublet sval)
{
	return arithmetic(fxs, tdouble, &sval, AOP_MINUS);
}
Flux *Flux::mul(Flux *fxs)
{
	return arithmetic(fxs, 0, nullx, AOP_MUL);
}
Flux *Flux::plus(Flux *fxs)
{
	return arithmetic(fxs, 0, nullx, AOP_PLUS);
}
Flux *Flux::div(Flux *fxs)
{
	return arithmetic(fxs, 0, nullx, AOP_DIV);
}
Flux *Flux::minus(Flux *fxs)
{
	return arithmetic(fxs, 0, nullx, AOP_MINUS);
}
Flux *Flux::dot(Flux *fxs, vector<vector<intt> > axis, sytet trans_order)
{
	Apply *apdot;
	Flux *fxo;
	vector<intt> axis_p, axis_s;
	intt axid[MX_DIM];
	intt ndim;

	if(qType != fxs->qType) throwFault(-1, "dot inconsistant type\n");
	
	TRCB(fxTcr)->bdot(this, fxs, axis, trans_order);

	axis_p = axis.at(0); 
	axis_s = axis.at(1);
	
	if(axis_p.size() != axis_s.size()) {
		apdot = new(fxTcr)ApDot2(TRACER(fxTcr), this, fxs, &axis_p, &axis_s, trans_order, ndim, axid);
		fxo = new(fxTcr)Flux(fxTcr, axid, ndim, qType, apply);
		((ApDot2 *)apdot)->dotOut = fxo;
	} else {
		apdot = new(fxTcr)ApDot(TRACER(fxTcr), this, fxs, &axis_p, &axis_s, trans_order, ndim, axid);
		fxo = new(fxTcr)Flux(fxTcr, axid, ndim, qType, apply);
		((ApDot *)apdot)->dotOut = fxo;
	}
	backend_ps(this, fxs, fxo, apdot);
	return fxo;
}
Flux *Flux::matmul(Flux *fxs, sytet trans_order)
{
	intt axid[MX_DIM];
	intt ndim, joint;

	if(qType != fxs->qType) throwFault(-1, "dot inconsistant type\n");

	TRCB(fxTcr)->bmatmul(this, fxs, trans_order);

	ndim = ApMatmul::matmul_out_shape(this, fxs, axid, trans_order, joint);
	Flux *fxo = new(fxTcr)Flux(fxTcr, axid, ndim, qType, apply);
	ApMatmul *apmul = new(fxTcr)ApMatmul(TRACER(fxTcr), this, fxs, fxo, joint, trans_order);
	
	backend_ps(this, fxs, fxo, apmul);
	return fxo;
}
vector<Flux *> *Flux::devide(intt axid[MX_DIM], intt ndim, intt nby, intt axis, bool ustack)
{
	vector<Flux *> *rsp = new vector<Flux *>;
	ApSplit *aps;
	Flux *fxo;
	intt i;

	if(scaleout > 0 && axis == 0) throwFault(-1, "variable rank and split axis error\n");

	aps = new(fxTcr)ApSplit(TRACER(fxTcr), this, nby, axis, ustack);
	referenced(aps);

	for(i = 0;i < nby; i++) {
		fxo = new(fxTcr)Flux(fxTcr, axid, ndim, qType, apply);
		aps->outputs(fxo);
		fxo->projected(aps);
		rsp->push_back(fxo);
		//fnisqgt.if(TRACER(fxTcr)->prompt) fxo->instTens(false);
		if(scaleout > 0) fxo->scaleout = scaleout;
		if(fxo->fshape[0] < fxo->scaleout) throwFault(-1, "devide scale out error %d %d\n", fxo->fshape[0], fxo->scaleout);
	}
	aps->opEnding();
	TRACER(fxTcr)->listv(rsp);

	backend_nstep(this, nullptr, nullx, aps);

	return rsp;
}
vector<Flux *> *Flux::split(intt nby, intt axis)
{
	intt axid[MX_DIM];
	intt i;
	
	if(axis < 0 || axis > fdim -1) throwFault(-1, "axis out range\n");
	if(fshape[axis] % nby) throwFault(-1, "devide fault %d by %d\n", fshape[axis], nby);

	TRCB(fxTcr)->bsplit(this, nby, axis);

	for(i = 0;i < fdim; i++) {
		if(i == axis) axid[i] = fshape[i] / nby;
		else axid[i] = fshape[i];
	}
	return devide(axid, fdim, nby, axis, false);
}
vector<Flux *> *Flux::unstack(intt axis)
{
	intt axid[MX_DIM];
	intt i, j;

	if(axis < 0 || axis > fdim - 1) throwFault(-1, "axis out range\n");

	TRCB(fxTcr)->bunstack(this, axis);

	for(i = j = 0;i < fdim; i++) {
		if(i != axis) axid[j++] = fshape[i];
	}
	return devide(axid, j, fshape[axis], axis, true);
}
bool Flux::realwiden(intt n)
{
	return TENSOR(quantum)->mxData->sizem2(n, false) > TENSOR(quantum)->mxData->maxmSize ? true : false;
}
intt Flux::copyt(void *psrc, ubytet tsrc, intt bsz) //bsz - ��ġ������, �ܺο��� ��ġ �����ɶ� ȣ��
{
	if(bsz > 0 && scaleout != bsz) {
		if(scaleout <= 0) throwFault(-1, "not various type\n");
		fshape[0] = FIRST_SHAPE_SCALE(this, bsz);
		scaleout = bsz;
		try {
			instTens(true);
		} catch(FaultObj fo) {
			throwFault(-1, "copy t\n%s", fo.fltmsg);
		}
	}//fnisqgt. else instTens(false);

	Univ uv(TYPED_WRITE_OP, tsrc);
	uv.cvuni = (longt)psrc;
	return ((Tensor *)quantum)->mxData->uniform(nullx, &uv);
}
intt Flux::copyf2(void *pdat, intt bsz, intt begin) //bsz - ��ġ������, �ܺο��� ��ġ �����ɶ� ȣ��
{
	if(bsz > 0 && scaleout != bsz) {
		if(scaleout <= 0) throwFault(-1, "not various type\n");
		fshape[0] = FIRST_SHAPE_SCALE(this, bsz);
		scaleout = bsz;
		try {
			instTens(true);
		} catch(FaultObj fo) {
			throwFault(-1, "copy t2\n%s", fo.fltmsg);
		}
	}//fnisqgt. else instTens(false);

	return TENSOR(quantum)->mxData->copyMemory(pdat, 0, begin, fxSize);
}
intt Flux::copyf(Flux *src)
{
	intt bsz = src->fshape[0];
	if(bsz > 0 && fshape[0] != bsz) {
		fshape[0] = bsz;
		try {
			instTens(true);
		} catch(FaultObj fo) {
			throwFault(-1, "copy f\n%s", fo.fltmsg);
		}
	}//fnisqgt. else instTens(false);

	return TENSOR(quantum)->mxData->copyMemory(src->begin_p(), 0, 0, fxSize);
}
void Flux::feedt(void *psrc, ubytet tsrc, intt sz) //�����̺�� ������� �ʴ´�.
{
	//TRCB(fxTcr)->bfeedt(this, psrc, tsrc, sz);//�����̺갡 ������� �ʵ��� build page�� ���� �ʰ�
	TRACER(fxTcr)->infeed(this, psrc, tsrc, sz);//�� �Լ��� ���� �����̺꿡 �����Ѵ�.
}
void Flux::feedf(void *pdat, intt sz) //�����̺�� ������� �ʴ´�.
{
	//TRCB(fxTcr)->bfeedf2(this, pdat, sz);//�����̺갡 ������� �ʵ��� build page�� ���� �ʰ�
	if(sz < 0) sz = fxSize;
	TRACER(fxTcr)->infeed(this, pdat, -1, sz);//�� �Լ��� ���� �����̺꿡 �����Ѵ�.
}
void Flux::feedf(Flux *src) //�����̺�� ������� �ʴ´�.
{
	//TRCB(fxTcr)->bfeedf(this, src);//�����̺갡 ������� �ʵ��� build page�� ���� �ʰ�
	//if(src->fxSize / src->fshape[0] != fxSize / fshape[0]) throwFault(-1, "inconsistant size\n");
	if(src->fxSize % (fxSize / fshape[0])) throwFault(-1, "inconsistant size\n");//nbatch2pseq������ ����

	//sizeCheck(src->fdim, src->fshape);
	TRACER(fxTcr)->infeed(this, src->begin_p(), -1, src->fxSize);//�� �Լ��� ���� �����̺꿡 �����Ѵ�.
	//resizing(src);
	//TENSOR(quantum)->mxData->inCopy(TENSOR(src->quantum)->mxData, 0);
}
Flux *Flux::feedf(ubytet fxtype, vinitfp vfp, const bytet *name)
{
	Flux *a = new(fxTcr)Flux(fxTcr, fshape, fdim, qType, fxtype, vfp, name);
	a->feedf(this);
	return a;
}
void Flux::resizing5(intt n, intt gid)//�׷��� �����忡�� arrange group�� �ϰ� ������¡�Ҷ� ȣ��
{
	if(ptrMastfx) return;//�� �÷����� �������� ����̰� ������ ���������� �Ŀ� ������ �����忡�� ȣ��Ǵ�
	//resizing 3���� �޸� �缳���� ����ǵ��� ����, ���⼭�� �����ϸ� resizing 3�� �ٸ� �����忡�� ���ÿ� 
	//����� �� �ִ�. -> �׷��� �����忡�� arrange group���� ��� �͵��� �װ͵��� ù��° ���� �ɸ��� ��� 
	//Ȯ���ϹǷ� �ٸ� �����忡�� reshape �����尡 ���ÿ� ������� ���� ���̹Ƿ� �̹����� ���� ���⼭��
	//instTens�����Ҷ� ptrMastfx�� �������� �����Ƿ� fastshape�����϶� �����߻��ϹǷ� ���� �ʴ´�.
	if(scaleout > 0) {//variable�÷����κ��� ��������� ����� �÷����϶��� ��ġȮ���Ѵ�.
		if(n < 0) throwFault(-1, "scalable link error1\n");
		if(scaleout != n) {
			fshape[0] = FIRST_SHAPE_SCALE(this, n);
			scaleout = n;
		}
	} else if(scaleout < 0) {//�� �÷����� reduce operation�� ����̸� ������¡ �� �ʿ����.
		if(DMATFX_GID(this) == gid) return;//�䱸 �׶��� ���̵� Ʋ����쿡�� �ش� �׶��忡 �Ҵ��Ѵ�. 
	} else if(n > 0) throwFault(-1, "scalable link error2\n");//n�� �Է��� Ȯ�� ��ġ �������̰� �Է��� �����Ϸ����ε� ����� �ƴ� ����̸� ����
	try {
		instTens(true, gid);//�̹��� ������ ������¡ �����ƴ� �ȵƴ� ��Ʈ������ ����� �̺��� ũ��
						//�׶��� ���̵� gid�� ������ ��Ʈ������ �޸𸮴� ���� �Ҵ���� �ʰ�
						//������ ���� ������¡ ���������� ��Ʈ������ ��ũ ������ ���ŵȴ�.
	} catch(FaultObj fo) {
		throwFault(-1, "resizing 5\n%s", fo.fltmsg);
	}
}
bool Flux::resizing(Flux *src)
{
	if(scaleout > 0) {
		if(src->scaleout <= 0) throwFault(-1, "scalable link error3\n");
		if(scaleout == src->scaleout) return false;//������//�����Ŀ� ������¡ ���฻��
		fshape[0] = FIRST_SHAPE_SCALE(this, src->scaleout);
		scaleout = src->scaleout;
		return true;//�����Ŀ� ������¡ �����Ұ�
	} else return false;//�����Ŀ� ������¡ ���฻��
}
void Flux::resizing2(Flux *src, const bytet *msg)
{
	//�� �Լ��� ȣ��Ǵ� ���� ��� �Է� ��ġ ����� ���� �������̹Ƿ� ����� �� �÷����� �����Ϸ����� �ƴϸ� ����
	if(scaleout == 0) {//scaleout�� 0���� �������� this�� mean�� ���� ���ེ ����� ����϶��̹Ƿ� ��ŵ
		if(fwAp) printf("pro ap: %d %p %p\n", ((Apply *)fwAp->vcontact)->apCode, this, (Apply *)fwAp->vcontact);
		if(bwLink) printf("back ap : %d %p %p\n", ((Apply *)bwLink)->apCode, this, (Apply *)bwLink);
		throwFault(-1, "%s %d scalable link error4\n", msg, scaleout);//
	}
	if(resizing(src)) {
		try {
			instTens(true);
		} catch(FaultObj fo) {
			throwFault(-1, "resizing 2\n%s", fo.fltmsg);
		}
	}
}
void Flux::resizing6(Flux *src)
{
	if(resizing(src)) {
		try {
			instTens(true);
		} catch(FaultObj fo) {
			throwFault(-1, "resizing 6\n%s", fo.fltmsg);
		}
	}
}
void Flux::resizing3(Flux *src) //������ �������߸� ���
{
	//Apreshape���� �Է�(src)����� ��������� �� �Լ��� ȣ��Ǵµ� �������� ����� this�� �����Ϸ����� 
	if(scaleout <= 0) throwFault(-1, "scalable link error5\n");//�ƴϸ� ����
	//frsz.�� �Լ��� ������ �������߸� ����ǰ� adges)���� �̸� ��ġȮ�� �������� resizing2�� �������
	//������ �����忡�� �� �Լ��� ȣ��ɶ� �Ʒ� ���� ù��° ȣ���� ������)���� ������¡ ��ŵ�Ǿ� �н�Ʈ 
	//�������϶� �����ɼ��־� �ι�° ���� �н�Ʈ �������϶��� ����� ���Ƶ� ������� �Ѵ�.(����-������ 
	//���� �߻��� ���� �ƴ�) ������ �н�Ʈ �������̸� �������� �޸𸮸� ���� �����Ǵµ� adges)���� ��������
	//������¡�� arrange �׷��� ��� ���� ù��° ���� ���Ľ����� ��� ��ġȮ��Ǳ⶧���� �������� �޸𸮵�
	//Ȯ������ ���ϼ��ִ�. ���� ���� ������ ������ �������� ���䷹����Ʈ �˻翡���� ������ ���� üũ�Ǿ�
	//����Ǿ� �������� Ȯ��� �޸𸮸� �����ϵ��� �н�Ʈ �������϶��� ����� ���Ƶ� ������� �Ѵ�.
	if(resizing(src) || TRACER(fxTcr)->fastshape) {
		try {
			instTens(true, -1, ptrMastfx);//reshape���� ȣ��ǹǷ� ptrMastfx����
		} catch(FaultObj fo) {
			throwFault(-1, "resizing 3\n%s", fo.fltmsg);
		}
	}
}
void Flux::resizing4(intt n)//������ ù��°�� ��ġ �����̸� ��ġ������ �ְų�(��, ��ġ������ �����Ϸ���
{//������ ����) �ƴϸ� ��ġ������ ������� �� �̰ų� ��ġ ������ �ٸ� ������ ���Ե� ���� �����̸� �־��� n����
	//���հ����� ����Ǿ� ���� ���̰ų� �϶� ȣ��
	if(scaleout > 0) scaleout = n;
	if(fshape[0] == n) return;
	fshape[0] = n;
	try {
		instTens(true);
	} catch(FaultObj fo) {
		throwFault(-1, "resizing 4\n%s", fo.fltmsg);
	}
}
void Flux::sizeCheck(intt ndim, intt axid[])
{
	if(fdim != ndim || memcmp(&fshape[1], &axid[1], (fdim - 1) * sizeof(intt))) throwFault(-1, "inconsistant shape\n");
}
Flux *Flux::reshape(vector<intt> axid)
{
	TRCB(fxTcr)->breshape(this, axid);

	intt none_def_axis = -1, mxsz = 1;
	for(intt i = 0;i < axid.size(); i++) {
		if(axid.at(i) < 0) {
			if(none_def_axis < 0) none_def_axis = i;
			else throwFault(-1, "reshape more than once non def dim\n");
		} else mxsz *= axid.at(i);
	}
	if(none_def_axis < 0) {
		if(fxSize != mxsz) throwFault(-1, "reshape size not fit\n");
	} else {
		if(fxSize % mxsz) throwFault(-1, "reshape not aligned\n");
		mxsz = fxSize / mxsz;
		axid[none_def_axis] = mxsz;
	}
	//if(scaleout > 0 && fshape[0] > axid[0] && axid[0] < scaleout) {
	//	throwFault(-1, "reshape scale out error %d %d %d\n", fshape[0], axid[0], scaleout);
	//}
	//reshape���� �޸𸮸� �����ϴ� ���� ����ҷ��� �Ʒ� this�� �츮�� ApReshape���� ������,����� ���� 
	//���縦 ���´�. ��� �������� �𸣰� �޸𸮸� �����ϸ� ê�� �׽�Ʈ ����� ������ �ʴ´�. ���߿� ����
	Flux *fxo = new(fxTcr)Flux(fxTcr, axid, qType, apply, nullptr, nullptr, TRACER(fxTcr)->fastshape ? this : nullptr);
	ApReshape *ars = new(fxTcr)ApReshape(TRACER(fxTcr), this, fxo);
	backend_ps(this, nullptr, fxo, ars);
	return fxo;
}
Flux *Flux::bypass(const bytet *msg) //����� ����Ǹ� �������ε� ����Ʈ�Ǵ� ����.
{
	if(TRACER(fxTcr)->pathPrint == 0 && TRACER(fxTcr)->bpPrint < -1) return this;

	TRCB(fxTcr)->bbypass(this);
	Flux *fxo = new(fxTcr)Flux(fxTcr, fshape, fdim, qType, apply, nullx, nullx);
	ApBypass *abp = new(fxTcr)ApBypass(TRACER(fxTcr), this, fxo, msg);

	backend_ps(this, nullptr, fxo, abp);
	return fxo;
}
Flux *Flux::partition(Tracer *gen_trc)
{
	TRCB(fxTcr)->bpartition(this);//���߿� gen_trc�� ���� ��� �ݿ�

	Trace *trc = gen_trc ? TRACER(gen_trc) : TRACER(fxTcr);
	Flux *fxo = new(trc)Flux(trc, fshape, fdim, qType, apply, nullx, nullx);
	
	ApPartition *apt = new(trc)ApPartition(trc, this, fxo);

	backend_ps(fxo, nullptr, nullptr, apt);//backward �������� �ʴ´�.
	fxo->partitionfx = 1;

	return fxo;
}
Flux *Flux::adjust(Flux *in)
{
	TRCB(fxTcr)->badjust(this, in);
	ApAdjust *adj = new(fxTcr)ApAdjust(TRACER(fxTcr), in, this);

	backend_ps(in, nullptr, this, adj);
	return this;
}
Flux *Flux::duplicate(Tracer *gen_trc)
{
	Flux *fxo = new(TRACER(gen_trc))Flux(TRACER(gen_trc), fshape, fdim, qType, variable, nullx, nullx);
	fxo->copyf(this);

	return fxo;
}
Flux *Flux::expand_dims(intt axis)
{
	vector<intt> axid;
	intt i;

	//TRCB(fxTcr)->bexpand_dims(this, axis);

	if(axis == -1) axis = fdim;
	for(i = 0;i < fdim; i++) {
		if(axis == i) axid.push_back(1);
		axid.push_back(fshape[i]);
	}
	if(axis == fdim) axid.push_back(1);
	return reshape(axid);
}
Flux *Flux::squeeze(intt axis)
{
	vector<intt> axid;
	intt i, j;

	//TRCB(fxTcr)->bsqueeze(this, axis);

	for(i = 0;i < fdim; i++) {
		if(axis >= 0) {
			if(axis == i) {
				if(fshape[i] > 1) throwFault(-1, "not one\n");
			} else axid.push_back(fshape[i]);
		} else if(fshape[i] != 1) axid.push_back(fshape[i]);
	}
	return reshape(axid);
}
Flux *Flux::transpose(vector<intt> axid)
{
	intt i = 0, txid[MX_DIM], nxid[MX_DIM];

	TRCB(fxTcr)->btranspose(this, axid);

	for(auto iter : axid) {
		txid[i] = iter;
		nxid[i++] = fshape[iter];
	}
	if(scaleout > 0 && txid[0] != 0) throwFault(-1, "variable rank and transpose axis error\n");

	Flux *fxo = new(fxTcr)Flux(fxTcr, nxid, i, qType, apply);
	ApTranspose *trs = new(fxTcr)ApTranspose(TRACER(fxTcr), this, fxo, txid);

	backend_ps(this, nullptr, fxo, trs);
	return fxo;
}
Flux *Flux::softmax(void)
{
	if(qType != tfloat && qType != tdouble) throwFault(-1, "type error\n");
	//if(qType == tint || qType == tshort) qt = tfloat;
	//else if(qType == tlong) qt = tdouble;
	//else qt = qType;
	TRCB(fxTcr)->bsoftmax(this);

	Flux *fxo = new(fxTcr)Flux(fxTcr, fshape, fdim, qType, apply);
	ApSoftmax *sft = new(fxTcr)ApSoftmax(TRACER(fxTcr), this, fxo);

	backend_ps(this, nullptr, fxo, sft);
	return fxo;
}
Flux *Flux::squaredDifference(Flux *fxs)
{
	if(qType != fxs->qType) throwFault(-1, "inconsistant type\n");

	TRCB(fxTcr)->bsquaredDifference(this, fxs);

	Flux *fxo = new(fxTcr)Flux(fxTcr, fshape, fdim, qType, apply);
	ApTwo *ap = new(fxTcr)ApTwo(TRACER(fxTcr), this, fxs, fxo, TWO_TYPE1, TWOF_SQDIFF);

	backend_ps(this, fxs, fxo, ap);//fsce.
	return fxo;
}
Flux *Flux::softmaxCrossEntropy(Flux *fxt)
{
	if(qType != fxt->qType) throwFault(-1, "inconsistant type\n");

	TRCB(fxTcr)->bsoftmaxCrossEntropy(this, fxt);

	Flux *fxo = new(fxTcr)Flux(fxTcr, fshape, fdim -1, qType, apply);
	ApSoftmaxCrossE *ap = new(fxTcr)ApSoftmaxCrossE(TRACER(fxTcr), this, fxt, fxo);

	backend_ps(this, fxt, fxo, ap);//fsce.
	return fxo;
}
Flux *Flux::sum(bool batch_sum)
{
	intt axid[MX_DIM];
	Flux *fxo;
	Apply *ap;

	TRCB(fxTcr)->bsum(this, batch_sum);

	if(batch_sum) {
		axid[0] = fshape[0];
		fxo = new(fxTcr)Flux(fxTcr, axid, 1, qType, apply);
		ap = new(fxTcr)ApBatchSum(TRACER(fxTcr), this, fxo, false);
	} else {
		axid[0] = 1;
		fxo = new(fxTcr)Flux(fxTcr, axid, 1, qType, apply);
		ap = new(fxTcr)ApSum(TRACER(fxTcr), this, fxo, false);
	}
	backend_ps(this, nullptr, fxo, ap, batch_sum ? 0 : -1);//������ 1���� �پ��� ���۷��̼��̸� ��ġȮ�����.

	return fxo;
}
Flux *Flux::mean(bool batch_sum)
{
	intt axid[MX_DIM];
	Flux *fxo;
	Apply *ap;

	TRCB(fxTcr)->bmean(this, batch_sum);

	if(batch_sum) {
		axid[0] = fshape[0];
		fxo = new(fxTcr)Flux(fxTcr, axid, 1, qType, apply);
		ap = new(fxTcr)ApBatchSum(TRACER(fxTcr), this, fxo, true);
	} else {
		axid[0] = 1;
		fxo = new(fxTcr)Flux(fxTcr, axid, 1, qType, apply);
		ap = new(fxTcr)ApSum(TRACER(fxTcr), this, fxo, true);
	}
	backend_ps(this, nullptr, fxo, ap, batch_sum ? 0 : -1);//������ 1���� �پ��� ���۷��̼��̸� ��ġȮ�����.

	return fxo;
}
void order_replace(intt axid[], intt i, intt v) //÷�ڸ� ������������ ����
{
	if(axid[i] < 0) axid[i] = v;
	else if(axid[i] < v) order_replace(axid, i + 1, v);
	else if(axid[i] > v) {
		order_replace(axid, i + 1, axid[i]);
		axid[i] = v;
	} else throwFault(-1, "dup index\n");
}
/*Flux *Flux::rsum(vector<intt> axid, bool mean, bool keep_dims)
{
	intt i = 0, j = 0, sum_xid[MX_DIM], nxid[MX_DIM];//not_xid[MX_DIM], k = 0, l = 0;

	//TRCB(fxTcr)->brmean(this, axid);���߿� ����

	for(auto iter : axid) {
		sum_xid[i++] = -1;
		order_replace(sum_xid, 0, iter);
	}
	for(i = 0;j < fdim; j++) {
		if(j == sum_xid[i]) {
			i++;
			if(keep_dims) nxid[l++] = 1;
		} else {
			not_xid[k++] = j;
			nxid[l++] = fshape[j];
		}
	}
	memset(sum_xid, 1, sizeof(intt) * MX_DIM);
	for(auto iter : axid) sum_xid[iter] = 0;
	for(i;i < fdim; i++) {
		if(sum_xid[i]) nxid[j++] = fshape[i];
		else if(keep_dims) nxid[j++] = 1;
	}
	Flux *fxo = new(fxTcr)Flux(fxTcr, nxid, j, qType, apply);
	ApReductSum *rsum = new(fxTcr)ApReductSum(TRACER(fxTcr), this, fxo, sum_xid, mean);

	backend_ps(this, nullptr, fxo, rsum);
	return fxo;
}*/
Flux *Flux::meanSquareError(Flux *fxt, bool mean)
{
	intt axid[MX_DIM];
	Flux *fxo;

	if(qType != fxt->qType) throwFault(-1, "inconsistant type\n");

	TRCB(fxTcr)->bmeanSquareError(this, fxt, mean);

	if(mean) {
		axid[0] = 1;
		fxo = new(fxTcr)Flux(fxTcr, axid, 1, qType, apply);
	} else fxo = new(fxTcr)Flux(fxTcr, fshape, fdim, qType, apply);
	
	ApMeanSquareE *ap = new(fxTcr)ApMeanSquareE(TRACER(fxTcr), this, fxt, fxo, mean);

	backend_ps(this, fxt, fxo, ap, mean ? -1 : 0);//fmse.//������ 1���� �پ��� ���۷��̼��̹Ƿ� ��ġȮ�����.
	return fxo;
}
Flux *Flux::actf(intt actf_op, floatt alpha)
{
	if(actf_op >= ACTF2_PRELU) return prelu(0.25);//https://arxiv.org/pdf/1502.01852.pdf

	TRCB(fxTcr)->bactf(this, actf_op);

	if(TRACER(fxTcr)->relualpha > 0) alpha = TRACER(fxTcr)->relualpha;
	Flux *fxo = new(fxTcr)Flux(fxTcr, fshape, fdim, qType, apply);
	ApActf *ap = new(fxTcr)ApActf(TRACER(fxTcr), this, fxo, actf_op, alpha);

	backend_ps(this, nullx, fxo, ap);
	return fxo;
}
Flux *Flux::tanh(void)
{
	return actf(ACTF_TANH);
}
Flux *Flux::relu(void)
{
	return actf(ACTF_RELU);
}
Flux *Flux::lrelu(floatt alpha)
{
	return actf(ACTF_RELU, alpha);
}
Flux *Flux::sigmoid(void)
{
	return actf(ACTF_SIGM);
}
Flux *Flux::prelu(floatt iv)
{
	TRCB(fxTcr)->bprelu(this, iv);

	Flux *fxo = new(fxTcr)Flux(fxTcr, fshape, fdim, qType, apply);
	TRACER(fxTcr)->directx(1);
	Flux *fxs = new(fxTcr)Flux(fxTcr, fshape, fdim, qType, trainable);
	TRACER(fxTcr)->directx(0);
	fxs->fill(iv);
	ApActf2 *ap = new(fxTcr)ApActf2(TRACER(fxTcr), this, fxs, fxo, ACTF_PRELU);

	backend_ps(this, fxs, fxo, ap);
	return fxo;
}
Flux *Flux::sqrt(void)
{
	return actf(MATH_SQRT);
}
Flux *Flux::log(void)
{
	return actf(MATH_LOG);
}
Flux *Flux::embedding_lookup(Flux *fxi)
{//this�� �Ӻ��� ���̺�, fxi�� Ÿ��
	intt axid[MX_DIM], i;

	TRCB(fxTcr)->bembeding_lookup(this, fxi);

	for(i = 0;i < fxi->fdim; i++) {
		axid[i] = fxi->fshape[i];
	}
	axid[i++] = fshape[fdim - 1];

	Flux *fxo = new(fxTcr)Flux(fxTcr, axid, i, qType, apply);
	ApEmbedding *ap = new(fxTcr)ApEmbedding(TRACER(fxTcr), this, fxi, fxo);

	backend_ps(this, fxi, fxo, ap);
	return fxo;
}
Flux *Flux::one_hot(intt depth, doublet on_value, doublet off_value, intt axis, intt dtype)
{//axis������ depth ������ Ȯ���ϰ� ������ ���� �ش��ϴ� ����(��ġ)�� on_value���� ���� �����Ͱ� ��ġ�ߴ� ���� �����Ѵ�.
	intt axid[MX_DIM], i, j;
	if(axis < 0) axis = fdim;

	TRCB(fxTcr)->bone_hot(this, depth, on_value, off_value, axis, dtype);

	for(i = j = 0;i < fdim; i++) {
		if(axis == i) axid[j++] = depth;
		axid[j++] = fshape[i];
	}
	if(i == axis) axid[j++] = depth;
	Flux *fxo = new(fxTcr)Flux(fxTcr, axid, j, dtype < 0 ? qType : dtype, apply);
	ApOneHot *ap = new(fxTcr)ApOneHot(TRACER(fxTcr), this, fxo, on_value, off_value, axis, depth);

	backend_ps(this, nullx, fxo, ap);
	return fxo;
}
void Flux::boundSlice(intt n, intt code[], intt idx[], bool check)
{
	intt i = 0, j = 0;

	for(i = 0;i < n; i++, j = i / 3) {
		if(code[i] < 0) idx[i] = fshape[j] + code[i];
		else idx[i] = code[i];
		if(check) {
			if(idx[i] < 0) idx[i] = 0;
			else if(idx[i] >= fshape[j]) idx[i] = fshape[j] - 1;
		}
	}
}
Flux *Flux::slice(vector<vector<intt>> axis)
{
	intt axid[MX_DIM], code[MX_DIM * 3], slicer_idx[MX_DIM * 3];
	intt i = 0, j = 0, n;

	TRCB(fxTcr)->bslice(this, axis);

	for(auto idx : axis) {
		if(idx.size() == 0) {
			code[i++] = 0;
			code[i++] = -1;
			code[i++] = 1;
		} else if(idx.size() == 1) {
			code[i++] = idx.at(0);
			code[i++] = idx.at(0);
			code[i++] = 1;
		} else {
			code[i++] = idx.at(0);
			code[i++] = (idx.at(1) > 0 ? idx.at(1) -1 : idx.at(1));
			if(idx.size() > 2) code[i++] = idx.at(2);
			else code[i++] = 1;
		}
	}
	if(i / 3 < fdim) {
		for(j = 0;j < fdim; j++) {
			code[i++] = 0;
			code[i++] = -1;
			code[i++] = 1;
		}
	}
	n = fdim * 3;
	boundSlice(n, code, slicer_idx, false);
	
	for(i = j = 0;i < n;i += 3, j++) axid[j] = 1 + (slicer_idx[i + 1] - slicer_idx[i]) / slicer_idx[i + 2];

	Flux *fxo = new(fxTcr)Flux(fxTcr, axid, n / 3, qType, apply);
	ApSlice *ap = new(fxTcr)ApSlice(TRACER(fxTcr), this, fxo, code);

	intt vscale = scaleout;
	if(scaleout > 0 && fshape[0] != axid[0]) {//slice�� ù��° ������ ������ҵǾ�����
		throwFault(-1, "slice first rank can't change scale\n");
		//REDUCE_SCALE(vscale, axid);//�ҽ��� ù��° ������ ������ �ƿ����� ��ʴ�� �������� ����Ѵ�.
	}//else slice�� ù��° ������ �״���̸� �ҽ��� �������� �״�� ����
	backend_ps(this, nullx, fxo, ap, vscale); 
	return fxo;	
}
Flux *Flux::argmax(intt axis, sytet t_out)
{
	intt axid[MX_DIM], i, j;
	OneVar onev;

	TRCB(fxTcr)->bargmax(this, axis);

	if(axis < 0) axis = fdim - 1;
	for(i = j = 0;i < fdim; i++) {
		if(axis == i) continue;
		axid[j++] = fshape[i];
	}
	onev.idxOne[0] = axis;
	if(t_out < 0) t_out = qType;
	Flux *fxo = new(fxTcr)Flux(fxTcr, axid, j, t_out, apply);
	ApArgmax *ap = new(fxTcr)ApArgmax(TRACER(fxTcr), this, fxo, axis);

	backend_ps(this, nullx, fxo, ap);
	return fxo;
}
Flux *Flux::vmax(intt axis)
{
	intt axid[MX_DIM], i, j;

	TRCB(fxTcr)->bvmax(this, axis);

	if(axis < 0) axis = fdim - 1;
	for(i = j = 0; i < fdim; i++) {
		if(axis == i) continue;
		axid[j++] = fshape[i];
	}
	Flux *fxo = new(fxTcr)Flux(fxTcr, axid, j, qType, apply);
	ApVMax *ap = new(fxTcr)ApVMax(TRACER(fxTcr), this, fxo, axis);

	backend_ps(this, nullx, fxo, ap);
	return fxo;
}
Flux *Flux::equal(Flux *fxs, doublet cmpv, bool cscalr, bool eq)
{
	Flux *fxo = new(fxTcr)Flux(fxTcr, fshape, fdim, qType, apply);
	OneVar onev;
	
	TRCB(fxTcr)->bequal(this, fxs, cmpv, cscalr, eq);

	onev.idxOne[0] = eq;
	if(cscalr) {
		onev.idxOne[1] = 1;
		univ_val_type(&onev.idxOne[2], &cmpv, qType, doublet, 0);
	} else onev.idxOne[1] = 0;

	ApOne *ap = new(fxTcr)ApOne(TRACER(fxTcr), this, fxs, fxo, &onev, false, 2, -1, AOP_EQUAL, 0, -1, 0, PDC, 0);

	backend_ps(this, fxs, fxo, ap);
	return fxo;
}
Flux *Flux::clipValue(doublet low, doublet high)
{
	TRCB(fxTcr)->bclipval(this, low, high);

	Flux *fxo = new(fxTcr)Flux(fxTcr, fshape, fdim, qType, apply);
	OneVar onev;

	*(doublet *)&onev.idxOne[0] = low;
	*(doublet *)&onev.idxOne[2] = high;

	ApOne *ap = new(fxTcr)ApOne(TRACER(fxTcr), this, nullx, fxo, &onev, 1, 2, 2, AOP_TYPE1, TYPE1_CLIP, -1, 0, PDC, 0);

	backend_ps(this, nullx, fxo, ap);
	return fxo;
}

void Flux::dstrw(const bytet dstr[])
{
	Univ uv(DSTR_WRITE_OP, 0);

	//TRCB(fxTcr)->bdstrw(this, dstr, strlen(dstr) +1);//���� �����ҷ��� �� �Լ��� ����� �� �÷����� �ٽ� feedf�� �Ű������� ȣ���Ͽ� feed�Ѵ�.

	//fnisqgt.instTens(true);
	uv.cvuni = (longt)dstr;
	TENSOR(quantum)->mxData->uniform(nullx, &uv);
}
Flux *Flux::overwrite(Flux *tar)
{
	TRCB(fxTcr)->boverwrite(this, tar);

	Flux *fxo = new(fxTcr)Flux(fxTcr, { 1 }, qType, apply);//�׷����� �����ϱ����� �±׿�Ȱ
	ApOverwrite *ap = new(fxTcr)ApOverwrite(TRACER(fxTcr), this, tar, fxo);

	backend_ps(this, tar, fxo, ap, -1);//fxo�� �ܼ��� �׷����� �����ϱ����� �±׿�Ȱ�ӷ� off��Ų��.

	return fxo;
}
void Flux::resetData(sytet gpu) //gpu: 1 , cpu: -1, mode: 0 
{//gpu device arrange ����� ���¿��� ȣ��Ǿ� �Ѵ�.
	TENSOR(quantum)->mxData->resetMemory(gpu);
}
void Flux::dumpToGrad(void) //gpu device arrange ����� ���¿��� ȣ��Ǿ� �Ѵ�.
{
	TENSOR(quantum)->mxGrad->inCopy(((Tensor *)quantum)->mxData, 0);
}
void Flux::resetGrad(void) //gpu device arrange ����� ���¿��� ȣ��Ǿ� �Ѵ�.
{
	TENSOR(quantum)->mxGrad->resetMemory(0);
}
void Flux::shape(void)
{
	intt i = 0;

	printf("(");
	for(;i < fdim - 1; i++) {
		printf("%d, ", fshape[i]);
	}
	printf("%d)\n", fshape[i]);
}
void *Flux::begin_p(intt off)
{
	return TENSOR(quantum)->mxData->begin_p(off);
}
void *Flux::begin_wp(intt off)
{
	return TENSOR(quantum)->mxData->begin_wp(off);
}
void *Flux::end_p(void)
{
	return TENSOR(quantum)->mxData->end_p();
}
void *Flux::read_p(vector<intt> axid, intt *rsz, intt n)
{
	intt i = 0, xid[MX_DIM], irank;

	for(auto iter : axid) xid[i++] = iter;
	irank = i;
	for(; i < fdim; i++) xid[i] = 0;

	if(xid[0] > fshape[0]) throwFault(-1, "read p 0 rank size over %d %d\n", xid[0], fshape[0]);
	//n�� ��õǸ� ��õ� �� ��ũ�� n������������ ����� rsz�� ���ϵȴ�.
	return TENSOR(quantum)->mxData->read_p(xid, irank, rsz, n);
}
//wsz - ������ ��ũ�� ����� ��ü ������ ������
void Flux::write_p(vector<intt> axid, void *dat, intt wsz)
{
	intt i = 0, xid[MX_DIM], irank;

	for(auto iter : axid) xid[i++] = iter;
	irank = i;
	for(; i < fdim; i++) xid[i] = 0;

	if(xid[0] > fshape[0]) throwFault(-1, "write p 0 rank size over %d %d\n", xid[0], fshape[0]);
	//������ ������� ������ ������ ��õ� �� ��ũ�� ���� �������� �߸���.
	TENSOR(quantum)->mxData->write_p(xid, dat, irank -1, wsz);
}
//tar�� �������� ���Ͽ� src�� ���� ����, n - iseq���� n ���������� src���� ����
void Flux::howrite(Flux *src, intt iseq, intt n)
{
	intt rsz = 0;
	void *p;

	resizing6(src);
	for(intt i = 0; i < fshape[0]; i++) {
		p = src->read_p({ i }, &rsz, n);
		write_p({ i, iseq }, p, rsz);
	}
}
//��ġ���� iseq��° ���������� �����㿩 tar�� �����Ѵ�, �̶� tar�� [batch, 1, feat] �̴�.
//n�� 1 �̻��̸� n�� �������� �����Ѵ�. �̶� ���� ȣ�⿡���� �������� �̾ �����Ѵٸ� iseq = iseq + n �� �ؾ��Ѵ�.
void Flux::horead(bytet *tar, intt iseq, intt n)
{
	intt rsz = 0, wsz;
	void *p;

	read_p({ 0 }, &rsz, n);//(1�� �������� ���� ����) * n
	wsz = sizefx(1) * rsz;//type size * rsz

	for(intt i = 0; i < fshape[0]; i++, tar += wsz) {
		p = read_p({ i, iseq }, &rsz, n);
		memcpy(tar, p, wsz);
	}
}
#define RET_T_VAL(dtp, qtp, ptr, i) {\
	switch(qtp) {\
		case tshort:\
			return (dtp)*((shortt *)ptr + i);\
		case tfloat:\
			return (dtp)*((floatt *)ptr + i);\
		case tint:\
			return (dtp)*((intt *)ptr + i);\
		case tlong:\
			return (dtp)*((longt *)ptr + i);\
		case tdouble:\
			return (dtp)*((doublet *)ptr + i);\
	}\
}
doublet Flux::at_d(intt i)
{
	if(i == 0) cursorP = TENSOR(quantum)->mxData->begin_p();

	RET_T_VAL(doublet, qType, cursorP, i);
}
doublet Flux::at_d2(intt i)
{
	cursorP = TENSOR(quantum)->mxData->begin_p();

	RET_T_VAL(doublet, qType, cursorP, i);
}
void Flux::printo(sytet leaf_one, sytet width)
{//leaf_one�̸� �ǳ�(����) ������ 1�̸� �������� �ʰ� ������ ���������� ���� 2�̸� ��� ������ ���, 1�̸� �յ� ����  
	//fnisqgt.instTens(false);//�κй��� ���(�����Ͱ� ���� ���)
	TENSOR(quantum)->mxData->printo(leaf_one, width);
	cout << "\n";
}
void Flux::printg(sytet leaf_one, sytet width)
{
	//fnisqgt.instTens(false);
	TENSOR(quantum)->mxGrad->printo(leaf_one, width);
	cout << "\n";
}
void Flux::iprinto(intt i, bool nwl)
{
	TENSOR(quantum)->mxData->iprinto(i);
	if(nwl) cout << "\n";
}
void Flux::iprintg(intt i, bool nwl)
{
	TENSOR(quantum)->mxGrad->iprinto(i);
	if(nwl) cout << "\n";
}
Flux *Flux::arange(intt len)
{
	Univ uv(ARANGE_UNIV_OP, tint);

	TRCB(fxTcr)->barange(this, len);

	uv.cvuni = len;
	//fnisqgt.instTens(false);
	((Tensor *)quantum)->mxData->uniform(nullx, &uv);
	return this;
}
Flux *Flux::fill(doublet cv, Flux *fxs)
{
	TRCB(fxTcr)->bfill(this, &cv, tdouble, fxs);

	if(fxs) {//fxs�� ��õ����� fxs�� �Է����� this�� ������� �����Ͽ� ��ġ ����� ����Ǹ� �� ����� �ݿ��Ǿ� �� �÷�����
		ApFill *afill = new(fxTcr)ApFill(TRACER(fxTcr), fxs, this, &cv, tdouble);//fill�� ����ǰ� �Ѵ�. fxs��
		backend_ps(nullptr, fxs, this, afill);//�ܼ��� �׶��� ����ÿ� ��ġ����� this�� �ݿ��ϱ����� �Է����� �����Ѵ�.
	} else {
		//fnisqgt.instTens(false);
		((Tensor *)quantum)->mxData->fill(&cv, tdouble);
	}
	return this;
}
Flux *Flux::fill(floatt cv, Flux *fxs)
{
	TRCB(fxTcr)->bfill(this, &cv, tfloat, fxs);

	if(fxs) {
		ApFill *afill = new(fxTcr)ApFill(TRACER(fxTcr), fxs, this, &cv, tfloat);
		backend_ps(nullptr, fxs, this, afill);
	} else {
		//fnisqgt.instTens(false);
		((Tensor *)quantum)->mxData->fill(&cv, tfloat);
	}
	return this;
}
Flux *Flux::fill(longt cv, Flux *fxs)
{
	TRCB(fxTcr)->bfill(this, &cv, tlong, fxs);
	
	if(fxs) {
		ApFill *afill = new(fxTcr)ApFill(TRACER(fxTcr), fxs, this, &cv, tlong);
		backend_ps(nullptr, fxs, this, afill);
	} else {
		//fnisqgt.instTens(false);
		((Tensor *)quantum)->mxData->fill(&cv, tlong);
	}
	return this;
}
void Flux::expofill(intt exp_c)
{
	TRCB(fxTcr)->bexpofill(this, exp_c);
	((Tensor *)quantum)->mxData->expofill(exp_c);
}
Flux *Flux::expand_elen(intt n, intt axis) //������ ����, fill�� ���������� ��ó�� �������� ���.
{
	intt axid[MX_DIM];
	intt i, j, ndim = fdim +1;

	TRCB(fxTcr)->bexpand_elen(this, n, axis);

	for(i = j = 0;i < ndim; i++) {
		if(i == axis) axid[i] = n;
		else axid[i] = fshape[j++];
	}
	Flux *fxo = new(fxTcr)Flux(fxTcr, axid, ndim, qType, const_apply, nullx, nullx);
	((Tensor *)fxo->quantum)->mxData->expandelen(((Tensor *)quantum)->mxData, n, axis);

	return fxo;
}
void Flux::_randn(doublet m, doublet v)
{
	Univ uv(RANDN_UNIV_OP, 0);

	*(doublet *)&uv.cvuni = m;
	*(doublet *)&uv.cvuni2 = v;
	//fnisqgt.instTens(false);
	DMATFX(this)->uniform(TRACER(fxTcr)->trcCxt(DMATFX_GID(this)), &uv);
}
Flux *Flux::randn(doublet m, doublet v) //initializer�Լ����� �� �Լ��� ȣ���ϹǷ� �� �Լ���  
{
	TRCB(fxTcr)->brandn(this, m, v);//���� ������ �ʿ䰡 ���� 
	_randn(m, v);//initializer�Լ������� �� �Լ��� ȣ���Ѵ�. Ư�� �ҽ�tcr�� ������Ʈ ���� �����ϸ�
				//Ÿ��tcr������ ������Ʈ ���� �������� �����Ƿ� prand���� ����ġ�� �߻��Ѵ�.
	return this;
}
Flux *Flux::randu(doublet m, doublet v)
{
	Univ uv(RANDU_UNIV_OP, 0);

	TRCB(fxTcr)->brandu(this, m, v);

	*(doublet *)&uv.cvuni = m;
	*(doublet *)&uv.cvuni2 = v;
	//fnisqgt.instTens(false);
	DMATFX(this)->uniform(TRACER(fxTcr)->trcCxt(DMATFX_GID(this)), &uv);
	return this;
}
Flux *Flux::layer_dense(intt nout, intt actf_code, vinitfp vfp, const bytet *name)
{
	Flux *w, *b, *c, *d;
	intt axid[MX_DIM];
	bytet *s;

	axid[0] = fshape[fdim - 1];
	axid[1] = nout;
	w = flux(fxTcr, 2, axid, qType, trainable, vfp, name);
	axid[0] = nout;
	s = (bytet *)TRACER(fxTcr)->xalloc(strlen(name) + 5);
	sprintf(s, "%s_b", name);
	b = flux(fxTcr, 1, axid, qType, trainable, vfp, s);
	c = dot(w, { {fdim - 1}, {0} });
	d = c->plus(b);

	if(actf_code < 0) return d;
	if(actf_code == ACTF_RELU) d = d->relu();
	else if(actf_code == ACTF_TANH) d = d->tanh();
	else if(actf_code == ACTF_SIGM) d = d->sigmoid();
	else if(actf_code == ACTF_LRELU) d = d->lrelu();
	else if(actf_code == ACTF2_PRELU) d = d->actf(ACTF2_PRELU);
	else throwFault(-1, "invalid actf\n");

	return d;
}
Flux *Flux::layer_dense(intt nout, const bytet *actf, vinitfp vfp, const bytet *name)
{
	sytet actf_code;

	if(actf == nullx) actf_code = -1;
	else if(!strcmp(actf, "relu")) actf_code = ACTF_RELU;
	else if(!strcmp(actf, "tanh")) actf_code = ACTF_TANH;
	else if(!strcmp(actf, "sigmoid")) actf_code = ACTF_SIGM;
	else if(!strcmp(actf, "lrelu")) actf_code = ACTF_LRELU;
	else if(!strcmp(actf, "prelu")) actf_code = ACTF2_PRELU;
	else throwFault(-1, "invalid actf\n");

	return layer_dense(nout, actf_code, vfp, name);
}
Flux *Flux::layer_normal(const bytet *name)
{
	bytet s[1024];
	TRCB(fxTcr)->blayer_normal(this);
	Flux *fxo = new(fxTcr)Flux(fxTcr, fshape, fdim, qType, apply, nullx, nullx);
	TRACER(fxTcr)->directx(1);//����ġ�̹Ƿ� Ÿ���� apply�� �ټ������Ƿ� ���� ǥ���Ͽ� bflux���� �������� �ʰ��Ѵ�.
	sprintf(s, "%s_ln_gamma", name);
	Flux *gm = new(fxTcr)Flux(fxTcr, &fshape[fdim -1], 1, qType, trainable, Initializer::one, s);
	sprintf(s, "%s_ln_beta", name);
	Flux *be = new(fxTcr)Flux(fxTcr, &fshape[fdim - 1], 1, qType, trainable, Initializer::zero, s);
	TRACER(fxTcr)->directx(0);
	ApLayerNormal *aln = new(fxTcr)ApLayerNormal(TRACER(fxTcr), this, fxo, gm, be);

	gm->referenced(aln);
	be->referenced(aln);

	if(TRACER(fxTcr)->prompt || ((Apply *)aln)->loadOnExec) {
		gm->fill(1.0);
		//gm->printo();
		be->fill(0.0);
	}
	backend_ps(this, nullptr, fxo, aln);
	return fxo;
}

Flux *Flux::scoopup(intt slidey, intt slidex, intt stridey, intt stridex)
{
	ApScoop *apscoop;
	intt axid[MX_DIM], ndim;

	apscoop = new(fxTcr)ApScoop(TRACER(fxTcr), this, slidey, slidex, stridey, stridex, ndim, axid);
	Flux *fxo = new(fxTcr)Flux(fxTcr, axid, ndim, qType, variable);
	((ApScoop *)apscoop)->apOut = fxo;
	fxo->projected(apscoop);
	fxo->reentrance(1);//Ÿ�����κ����� ����� ������ �� fxo���� ���߰� �� fxo���� ������ ���۵ǰ� ����.

	if(TRACER(fxTcr)->prompt || apscoop->loadOnExec) {
		if(quantum == nullptr || TENSOR(quantum)->mxData->begin_p() == nullptr) throwFault(-1, "not intantiate\n");
		apscoop->forward(TRACER(fxTcr)->trcCxt(apscoop->oidground()), stridey, stridex);
	}
	return fxo;
}
void Flux::scoopup(intt stridey, intt stridex) //�� �Լ��� �����Ѱ� ����.
{
	((ApScoop *)bwLink)->forward(TRACER(fxTcr)->trcCxt(((ApScoop *)bwLink)->oidground()), stridey, stridex);
}
void Flux::minmax_normal(doublet minv, doublet maxv)
{
	OneVar onev;

	*(doublet *)&onev.idxOne[0] = minv;
	*(doublet *)&onev.idxOne[2] = maxv - minv;

	DMATFX(this)->mone(TRACER(fxTcr)->trcCxt(DMATFX_GID(this)), nullx, TENSOR(quantum)->mxData,
		2, &onev, AOP_ACTF, MINMAX_NORMAL, PDC3, 0);
}
void Flux::minmax(doublet &minv, doublet &maxv, sytet if_sync)
{
	Univ uv(MIN_MAX_V_OP, if_sync);//if_sync: 0 - not sync, 1 - dev to host load
	((Tensor *)quantum)->mxData->uniform(nullx, &uv);
	minv = *(doublet *)&uv.cvuni;
	maxv = *(doublet *)&uv.cvuni2;
}
void Flux::stdnormal(sytet if_sync)
{
	Univ uv(STD_NORMAL, if_sync);//if_sync: 0 - not sync, 1 - dev to host load, 2 - host to dev save, 3 - 1 & 2
	((Tensor *)quantum)->mxData->uniform(nullx, &uv);
}
//������ 0�� ������ ��ü �̵��Ͽ� ����� �����. (prev�� 0���̸� ���ϰ��[ G = A / (P + A) ]��
//1�̵Ǿ� ������ �ȵȴ�, ���� �����͸� ����� ��ȯ), �����Ҷ��� �Ѱ� �������� ���� ���� �򰡰�������
void Flux::xnormal(bool reverse, bool pavg, sytet if_sync)
{
	Univ uv(X_NORMAL, if_sync);//if_sync: 0 - not sync, 1 - dev to host load, 2 - host to dev save, 3 - 1 & 2
	*(sytet *)&uv.cvuni = reverse;
	*((sytet *)&uv.cvuni +1) = pavg;
	uv.cvuni2 = 0;
	((Tensor *)quantum)->mxData->uniform(nullx, &uv);
}
//�������� �������� �����Ҷ� �������� �������κ��� ȹ���Ͽ� ����, �� �Լ��� ���ӵǴ� �������� �����ϸ�
//�ڽ��� ������ ���� ������ ���� ���� �������� ���� ������ ����ϰ� �ǹǷ� ������ �������� �ŵ��ϸ�
//�����ǹǷ� �������κ��� �������� ȹ���Ͽ� �Լ��� ���ӵǴ� ������ �������� �����Ҷ� ���, �Ʒð�������
void Flux::xrnormal(Flux *origin, bool pavg, sytet if_sync)
{
	Univ uv(X_NORMAL, if_sync);//if_sync: 0 - not sync, 1 - dev to host load, 2 - host to dev save, 3 - 1 & 2
	*(sytet *)&uv.cvuni = 1;
	*((sytet *)&uv.cvuni + 1) = pavg;
	uv.cvuni2 = (longt)((Tensor *)origin->quantum)->mxData;
	((Tensor *)quantum)->mxData->uniform(nullx, &uv);
}
void Flux::sinpos(intt nseq)//, Flux *fxs)
{
	TRCB(fxTcr)->bsinpos(this, nseq, nullx);

	/*nseq�� ��ġ�� �ƴ� �������̹Ƿ� �����Ҷ��� �����Ҷ� ���̰� �ٸ� �� �����Ƿ� �ʿ����.
	if(fxs) {//fxs�� ��õ����� fxs�� �Է����� this�� ������� �����Ͽ� ��ġ ����� ����Ǹ� �� ����� �ݿ��Ǿ� �� �÷�����
		ApSinPos *aspos = new(fxTcr)ApSinPos(TRACER(fxTcr), fxs, this, nseq);//sinpos�� ����ǰ� �Ѵ�. fxs��
		backend_ps(nullptr, fxs, this, aspos);//�ܼ��� �׶��� ����ÿ� ��ġ����� this�� �ݿ��ϱ����� �Է����� �����Ѵ�.
	} else {*/
		Univ uv(SIN_POSITIONAL, 0);
		uv.cvuni = nseq;
		((Tensor *)quantum)->mxData->uniform(nullx, &uv);
	//}
}
intt Flux::groundid(bool grad)
{
	return grad ? GMATFX_GID(this) : DMATFX_GID(this);
}
Flux *AdamOptimizier::minimize(Flux *fxi, vector<Flux *> *weight_list)
{
	intt axid[2];

	TRCB(fxi->fxTcr)->bminimize(this, fxi);

	axid[0] = 1;
	Flux *fxo = new(fxi->fxTcr)Flux(fxi->fxTcr, axid, 1, fxi->qType, apply);
	ApAdamOpt *ap = new(fxi->fxTcr)ApAdamOpt(TRACER(fxi->fxTcr), fxi, fxo, rLning);

	fxo->termifx = true;
	fxi->referenced(ap);
	fxo->projected(ap);

	ap->minimize(weight_list);
	fxo->meanAfter = fxi->meanAfter;

	return fxo;
}

Flux *GradientDescentOptimizier::minimize(Flux *fxi, vector<Flux *> *weight_list)
{
	intt axid[2];

	TRCB(fxi->fxTcr)->bminimize(this, fxi);

	axid[0] = 1;
	Flux *fxo = new(fxi->fxTcr)Flux(fxi->fxTcr, axid, 1, fxi->qType, apply);
	ApSgdOpt *ap = new(fxi->fxTcr)ApSgdOpt(TRACER(fxi->fxTcr), fxi, fxo, rLning);

	fxo->termifx = true;
	fxi->referenced(ap);
	fxo->projected(ap);

	ap->minimize(weight_list);
	fxo->meanAfter = fxi->meanAfter;

	return fxo;
}
#include "anet.h"
namespace memput {
	namespace mp {
		AdamOptimizier *adam_optimizer(Tracer *tcr, floatt lr)
		{
			AdamOptimizier *opt = new(tcr)AdamOptimizier(lr);

			TRCB(tcr)->badam_optimizer(opt, lr);

			return opt;
		}
		GradientDescentOptimizier *gradient_descent_optimizer(Tracer *tcr, floatt lr)
		{
			GradientDescentOptimizier *opt = new(tcr)GradientDescentOptimizier(lr);

			TRCB(tcr)->bsgd_optimizer(opt, lr);

			return opt;
		}
		Flux *flux(Tracer *tcr, initializer_list<intt> axid, ubytet qtype, ubytet fxtype, vinitfp vfp, const bytet *name, intt iname)
		{
			Flux *a;
			bytet f_name[1024];
			if(name) {
				if(iname >= 0) {
					sprintf(f_name, "%s_%d", name, iname);
					name = f_name;
				}
				a = TRACER(tcr)->findfxns((bytet *)name);
				if(a) return a;
			}
			a = new(tcr)Flux(tcr, axid, qtype, fxtype, vfp, name);
			return a;
		}
		Flux *flux(Tracer *tcr, vector<intt> axid, ubytet qtype, ubytet fxtype, vinitfp vfp, const bytet *name)
		{
			Flux *a;
			if(name) {
				a = TRACER(tcr)->findfxns((bytet *)name);
				if(a) return a;
			} 
			a = new(tcr)Flux(tcr, axid, qtype, fxtype, vfp, name);
			return a;
		}
		Flux *flux(Tracer *tcr, Flux *src, ubytet fxtype, vinitfp vfp, const bytet *name)
		{
			Flux *a;
			if(name) {
				a = TRACER(tcr)->findfxns((bytet *)name);
				if(a) return a;
			} 
			a = new(tcr)Flux(tcr, src->fshape, src->fdim, src->qType, fxtype, vfp, name);
			a->copyf(src);
			return a;
		}
		Flux *flux(Flux *src, ubytet fxtype, vinitfp vfp, const bytet *name)
		{
			Flux *a;
			if(name) {
				a = TRACER(src->fxTcr)->findfxns((bytet *)name);
				if(a) return a;
			} 
			a = new(src->fxTcr)Flux(src->fxTcr, src->fshape, src->fdim, src->qType, fxtype, vfp, name);
			a->copyf(src);
			return a;
		}
		intt dstr_form(const bytet dstr[], intt axid[])
		{
			intt ndim = 0, i = -1, len = strlen(dstr), b = -1, c[MX_DIM];
			bool exist;

			memset(axid, 0x00, MX_DIM * sizeof(intt));
			while(1) {
				for(++i;i < len && dstr[i] != '[' && dstr[i] != ']'; i++) {
					bool s = false;
					for(; dstr[i] == ' ' || dstr[i] == '\t' || dstr[i] == '\n' || dstr[i] == ','; i++) s = true;
					if(s) {//�����̽�
						i--;
						if(b < 0) throwFault(-1, "syntax error\n");
						if(exist) c[b]++;
						exist = false;
					} else if('0' <= dstr[i] && dstr[i] <= '9') exist = true;
				}
				if(i == len) break;
				if(dstr[i] == '[') {
					b++; c[b] = 0;
					if(ndim == b) ndim++;
					exist = false;
				} else if(dstr[i] == ']') {
					if(exist == false) throwFault(-1, "empty array\n");
					c[b]++;
					if(axid[b]) {
						if(axid[b] != c[b]) throwFault(-1, "syntax error\n");
					} else axid[b] = c[b];
					b--;
				}
			}
			return ndim;
		}
		Flux *flux(Tracer *tcr, const bytet dstr[], ubytet qtype, ubytet fxtype, vinitfp vfp, const bytet *name)
		{
			intt axid[MX_DIM], ndim = 0;

			ndim = dstr_form(dstr, axid);
			Flux *a;
			if(name) {
				a = TRACER(tcr)->findfxns((bytet *)name);
				if(a) return a;
			}
			a = new(tcr)Flux(tcr, axid, ndim, qtype, fxtype, vfp, name);
			a->dstrw(dstr);
			return a;
		}
		Flux *flux(Tracer *tcr, intt ndim, intt axid[], ubytet qtype, ubytet fxtype, vinitfp vfp, const bytet *name)
		{
			Flux *a;
			if(name) {
				a = TRACER(tcr)->findfxns((bytet *)name);
				if(a) return a;
			} 
			a = new(tcr)Flux(tcr, axid, ndim, qtype, fxtype, vfp, name);
			//fnisqgt.a->instTens(true);
			return a;
		}
		Tracer *trace(sytet stepw, const bytet *name)
		{
			return new Trace(stepw, name);
		}
		intt Initializer::xavier(Flux *fx)
		{
			if(fx == nullx) return T_XAVIER_INIT;
			//longt v = fx->nJoint < 16 ? 16 : fx->nJoint;
			//fx->_randn(0, sqrt(1. / (doublet)v));
			longt v = fx->fdim == 1 ? fx->fshape[0] : fx->fshape[fx->fdim - 1] + fx->fshape[fx->fdim - 2];
			if(v <= 2) v = 2;
			fx->_randn(0, sqrt(2. / (doublet)v));
			return 0;
		}
		intt Initializer::he(Flux *fx)
		{
			if(fx == nullx) return T_HE_INIT;
			//longt v = fx->nJoint < 16 ? 16 : fx->nJoint;
			//fx->_randn(0, sqrt(1. / ((doublet)v / 2)));
			longt v = fx->fdim == 1 ? fx->fshape[0] : fx->fshape[fx->fdim - 2];
			if(v <= 2) v = 2;
			fx->_randn(0, sqrt(2. / (doublet)v));
			return 0;
		}
		intt Initializer::one(Flux *fx)
		{
			if(fx == nullx) return T_ONE_INIT;

			doublet cv = 1;
			((Tensor *)fx->quantum)->mxData->fill(&cv, tdouble);
			return 0;
		}
		intt Initializer::zero(Flux *fx)
		{
			if(fx == nullx) return T_ZERO_INIT;

			doublet cv = 0;
			((Tensor *)fx->quantum)->mxData->fill(&cv, tdouble);
			return 0;
		}
		Generic *generic(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
			sytet af, floatt lr, const bytet *name)
		{
			return new(ingate == nullptr ? targate->fxTcr : ingate->fxTcr)Generic(ingate == nullptr ? targate->fxTcr : ingate->fxTcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, af, lr, name);
		}
		Generic *stepwise(Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim,
			sytet af, floatt lr, const bytet *name, bool auto_encoder)
		{
			return new(ingate->fxTcr)Generic(ingate->fxTcr, ingate, outsz, latent_sz, indiscret, embedim, af, lr, name, auto_encoder);
		}
		Generic *generic(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
			sytet af, floatt lr, const bytet *name)
		{
			return new(tcr)Generic(tcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, af, lr, name);
		}
		Generic *stepwise(Tracer *tcr, Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim,
			sytet af, floatt lr, const bytet *name, bool auto_encoder)
		{
			return new(tcr)Generic(tcr, ingate, outsz, latent_sz, indiscret, embedim, af, lr, name, auto_encoder);
		}
		Algol *algol(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
			bool contraction, sytet af, floatt lr, const bytet *name)
		{
			return new(ingate->fxTcr)Algol(ingate->fxTcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, contraction, af, lr, name);
		}
		Stratus *stratus(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
			sytet af, floatt lr, const bytet *name)
		{
			return new(ingate->fxTcr)Stratus(ingate->fxTcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, af, lr, name);
		}
		void BoostMemput(intt gid, sytet boot_if)
		{
			glb::rsc::BoostMemput(gid, boot_if);
		}
		void throwFault(intt tflt, const char *fmt, ...)
		{
			va_list ap;
			FaultObj e;
			bytet *p = e.fltmsg;

			e.fltcd = tflt;
			sprintf(p, ERR_MSG_HEAD, tflt);
			p += strlen(p);

			va_start(ap, fmt);
			vsprintf(p, fmt, ap);

			e.fltoff = strlen(e.fltmsg);
			printf("%s", e.fltmsg);

			va_end(ap);
			
			throw(e);
		}
		void printo(vector<Flux *> *fxl)
		{
			intt i = 0, n = fxl->size();

			cout << "\n[";
			for(auto pfx : *fxl) {
				if(i++) cout << ",";
				cout << "array(";
				pfx->printo();
				cout << ")";
				if(i != n) cout << "\n";
			}
			cout << "]\n";
		}
		void printg(vector<Flux *> *fxl)
		{
			intt i = 0, n = fxl->size();

			cout << "\n[";
			for(auto pfx : *fxl) {
				if(i++) cout << ",";
				cout << "array(";
				pfx->printg();
				cout << ")";
				if(i != n) cout << "\n";
			}
			cout << "]\n";
		}
		void dumpToGrad(vector<Flux *> *fxl) //gpu device arrange ����� ���¿��� ȣ��Ǿ� �Ѵ�.
		{
			for(auto pfx : *fxl) {
				pfx->dumpToGrad();
			}
		}
		void resetGrad(vector<Flux *> *fxl) //gpu device arrange ����� ���¿��� ȣ��Ǿ� �Ѵ�.
		{
			for(auto pfx : *fxl) {
				pfx->resetGrad();
			}
		}
		Flux *combine(Trace *tcr, vector<Flux *> *fxl, intt axid[], intt ndim, intt nplus, intt axis, sytet qt, bool cstack)
		{
			ApConcat *acc;
			Flux *fxo;
			bool all_inst = true;

			fxo = new(tcr)Flux(tcr, axid, ndim, qt, apply);
			acc = new(tcr)ApConcat(tcr, fxo, nplus, axis, cstack);
			//fnisqgt.if(tcr->prompt) fxo->instTens(false);

			Flux *fx = fxl->at(0);
			intt dim = fx->fshape[axis];
			bool parity = 1;
			for(auto fxi : *fxl) {
				fxi->referenced(acc);
				acc->inputs(fxi);
				if(fxi->fshape[axis] != dim) parity = 0;
				if(fxi->quantum == nullptr || TENSOR(fxi->quantum)->mxData->begin_p() == nullptr) all_inst = false;
				if(fxi->scaleout > 0) fxo->scaleout = fxi->scaleout;
				if(fxo->fshape[0] < fxo->scaleout) throwFault(-1, "combine scale out error %d %d\n", fxo->fshape[0], fxo->scaleout);
			}
			if(fxo->scaleout > 0 && axis == 0) throwFault(-1, "variable rank and combine axis error\n");

			fxo->projected(acc);
			acc->parityParty = parity;
			acc->opEnding();

			if(tcr->prompt) {
				Matrixr *mx = nullx;
				if(all_inst == false) throwFault(-1, "not intantiate\n");
				acc->forward(tcr->trcCxt(acc->oidground()), mx);
			}
			return fxo;
		}
		Flux *concat(vector<Flux *> *fxl, intt axis)
		{
			Flux *first = nullptr;
			intt axid[MX_DIM];
			intt n = 0, cat_dim = 0;
			Trace *tcr = nullptr;

			for(auto fxi : *fxl) {
				if(n++ == 0) {
					first = fxi;
					tcr = TRACER(first->fxTcr);
					for(intt j = 0;j < fxi->fdim; j++) axid[j] = fxi->fshape[j];
				}
				if(first->fdim != fxi->fdim) throwFault(-1, "inconsistancy dim\n");
				if(first->qType != fxi->qType) throwFault(-1, "inconsistancy type\n");
				for(intt j = 0;j < fxi->fdim; j++) {
					if(j != axis && axid[j] != fxi->fshape[j]) throwFault(-1, "inconsistancy shape\n");
					if(j == axis) cat_dim += fxi->fshape[j];
				}
			}
			TRCB(first->fxTcr)->bconcat(fxl, axis);

			axid[axis] = cat_dim;
			return combine(tcr, fxl, axid, first->fdim, n, axis, first->qType, false);
		}
		Flux *concat(initializer_list<Flux *> fxl, intt axis)
		{
			vector<Flux *> vfxl;
			for(auto fxi : fxl) vfxl.push_back(fxi);
			return concat(&vfxl, axis);
		}
		Flux *stack(vector<Flux *> *fxl, intt axis)
		{
			Flux *first = nullptr;
			intt axid[MX_DIM];
			intt n = 0;
			Trace *tcr = nullptr;

			for(auto fxi : *fxl) {
				if(n++ == 0) {
					intt i = 0, j = 0;
					first = fxi;
					tcr = TRACER(first->fxTcr);
					for(;i < first->fdim +1; i++) {
						if(i != axis) axid[i] = first->fshape[j++];
					}
				}
				if(first->fdim != fxi->fdim) throwFault(-1, "inconsistancy dim\n");
				if(first->qType != fxi->qType) throwFault(-1, "inconsistancy type\n");
				for(intt j = 0;j < fxi->fdim; j++) {
					if(first->fshape[j] != fxi->fshape[j]) throwFault(-1, "inconsistancy shape\n");
				}
			}
			TRCB(first->fxTcr)->bstack(fxl, axis);

			axid[axis] = n;
			return combine(tcr, fxl, axid, first->fdim +1, n, axis, first->qType, true);
		}
		Flux *stack(initializer_list<Flux *> fxl, intt axis)
		{
			vector<Flux *> vfxl;
			for(auto fxi : fxl) vfxl.push_back(fxi);
			return stack(&vfxl, axis);
		}
		void lbackward(vector<Flux *> *fxl, void *tcxt)
		{
			Apply *first_bwa = nullptr, *bwa;
			Matrixr *mx = nullx;

			for(auto pfx : *fxl) {
				bwa = ((Apply *)pfx->bwAp)->dFanOut();
				if(bwa) {
					if(first_bwa) {//bwa�� ���, �ٸ� �����尡 ����� ����
					} else first_bwa = bwa;
				}
			}
			if(first_bwa) first_bwa->backward(tcxt ? (TContext *)tcxt : first_bwa->apTcr->trcCxt(first_bwa->oidground()), mx);
		}
		void *mp_pointer_t(void)
		{
			return rsc::srutra;
		}
	}
}