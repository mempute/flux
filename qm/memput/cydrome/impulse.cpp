
#include "impulse.h"

thrret ThrRunisle(thrarg arg)
{
	Isleloop *isle = (Isleloop *)arg;

	if(isle->tadaptisle) return 0;

	isle->tadaptisle = 1;
	isle->runisle();

	return 0;
}

Impulse::~Impulse()
{
	for(Isleloop *isle = islelist;isle; isle = isle->ptrRight) isle->rmloop();
	for(Dynagen *dgen = dgenlist;dgen; dgen = dgen->ptrRight) dgen->dgenremove();

	ReleaseSelPage(qcimp);
	CLOSE_MUT_(mutimp);
	CLOSE_LSEM_(semimp);
	rsc::cydrome->putcydid(cydimp);
}
void Impulse::recording(void)
{
	for(Isleloop *isle = islelist;isle; isle = isle->ptrRight) isle->isletrc->saveWeight();
}
void Impulse::retrain(void)
{
	for(Isleloop *isle = islelist; isle; isle = isle->ptrRight) isle->alignoper = 0;
}
void Impulse::regiisle(Isleloop *isle)
{
	ntrainisle++;
	APPEND_LIST(islelist, isle);
}
void Impulse::regidgen(Dynagen *dgen)
{
	APPEND_LIST(dgenlist, dgen);
}
void Impulse::coupleisleThr(void)
{
	for(Isleloop *isle = islelist;isle; isle = isle->ptrRight) {
		xthread_create((void *)ThrRunisle, isle);
	}
}

Flux *Impulse::train(intt *n_train) //������ Ȯ���� ���޽� �Ҽ� ���̳����� dgenloss�� �����Ѵ�.
{
	Flux *tloss;
	if(tadapt < 2) {
		if(tadapt == 0) coupleisleThr();
		dgenlist->ptrLeft->dumyloss();//�ʱ⿡ �ν��� ������� �ʾ� �ؿ��� tloss�� �η� �����Ǵ� ���� ����
		if(impTcr->concurtrain) {//�����н��̸� ������������ ������ �����Ƿ� �н��Ϸ�üũ ����� ���� isle��
			for(SyncInvoke *s = synchor;s; s = s->ptrRight) {//�����Ƿ� ���⼭ �ʱ��� ���ش�.
				s->invokeisle->decreaseProtect();
			}
		}
		tadapt = 2;
	}
	if(runPredict) {//�߰� ���ڴ�isle�� �н��� �Ϸ���� ���� �߰��� ������ �����ϰ� �ٽ� �н���
		runPredict = 0;//�����ϴ� ��� ������������ ��ǥ���� ������ ��� Ŀ��isle�� ���۷��尡  
		retrain();//���ĵǾ� ���ĺ��ʹ� ieij)���� Ŀ��isle�� �Է°� Ÿ�� ������ ���Ե���������
				//���� �ɸ��µ� ���ڴ�isle�� �н��� ������ �ʾ����Ƿ� Ŀ��isle�� �Է��� ���Ե���
				//�����Ƿ� ��ü ���� �ɸ���. ���� �̰�� �ٽ� �����Ѵ�. �н��߰��� ������ ����
	}			//�ϴ� ��� �н��� ���Ե� ���� ���̶u���� ieie)���� ���ŵȴ�.
	predwithloss = 1;
	trainstep = init_trainstep;
	if(impTcr->concurtrain > 1) {//��ü �����н�(�ܰ躰 �н��Ϸ��� �����尡 �ƴ�) 2�ܰ��̸� 
		finalSync = 1;				//�� ���� ��ü isle �н� ���� �� ����
		itrainisle = (trainstep == 1 ? 0 : idonetrain);//2�ܰ� �н��̸� iril)����
	} //2�ܰ� �н� �Ϸ���� �ʴ� ���ڴ��� �н� ī��Ʈ �ǹǷ� �ʱⰪ���� �н� �Ϸ�� isle������ �����Ѵ�.
	for(SyncInvoke *s = synchor;s; s = s->ptrRight) {//sockid�� 1�ΰ��� Ŀ�þ����϶� 
		//Ÿ���ʸ� ���� �����ϴ� �ϴ� ����̹Ƿ� Ÿ���� �Է����� �����Ѵ�.
		s->invokeisle->inject(s->sockid == 1 ? s->invokeisle->exttargate : s->invokeisle->extingate, s->invokeisle->exttargate, syncIndice, s->sockid);
	}
	syncIndice++;
	if(finalSync) waittraining();//inject������ �÷ο� ��Ʈ�� �ǹǷ� ��� �ʿ������(�������̸�
	//������ ���� ���� �����Ҷ� ���ȴ�) �����н� 2�ܰ��̸� �� ���� ��isle�� �Ϸ�ɶ����� ���
	dgenlist->ptrLeft->predictv(&tloss);//��ǥ�� ������ dgen������ ����, �ʿ��ϸ� �� dgen���� Ȯ��
	if(init_trainstep != -1) {//2�ܰ� �н��� ���۵��� �ʾ�����(1�ܰ� �н� �������̸�)
		if(ntrainisle == idonetrain) {//1�ܰ� �н��Ϸ� ������ 2�ܰ� �н� ����.
			init_trainstep = -1;//2�ܰ� �н� ǥ��(���ĺ��ʹ� ���ڴ��� �н� �����)
			idonetrain -= ndecisle;//���ڴ� isle������ ���־� 2�ܰ� �н��� ���ڴ� ������ ī��Ǹ� ��ü �н� �Ϸ�üũ�ǰ� �Ѵ�.
		}
	}
	tloss->printo();
	if(n_train) *n_train = ntrainisle - idonetrain;
	return tloss;//���� isle�� �սǸ� ����ǹǷ� ����isle�� �����ϱ������� �⺻���� 0�� ��µȴ�.
}
Flux *Impulse::predict(Flux **ploss) //�������� Ȯ���� ���޽� �Ҽ� ���̳����� dgenpred�� �����Ѵ�.
{
	sytet sid;
	if(tadapt == 0) {
		coupleisleThr();
		tadapt = 1;
	}
	//�н��߰��� ������ �����ϴ� ��� �н��� ���Ե� ���� ���̶u���� ieie)���� ���ŵȴ�.
	if(ploss) predwithloss = 1;
	else predwithloss = 0;
	runPredict = 1;
	trainstep = 0;
	finalSync = 0;
	syncIndice++;
	for(SyncInvoke *s = synchor;s; s = s->ptrRight) {
		sid = s->sockid;
		if(predwithloss == 0) {//�򰡶� �ν� ��� ���� ���.
			if(sid == 1) continue;//Ÿ�� ������ �ƴϸ� Ÿ���� ������ ���� �ʴ´�.
			//else if(s->sockid == -2) sid = -1;//���� ���� �����̸� �Է����� �����ϰ� �Ѵ�.
			sid = -1;//���Ͼ��̵� 0�̴� -2(���� ���� ����)�̴� -1���Ͽ� ijed)���� �Է¸� ���Եǰ� �Ѵ�.
		}//else �򰡶� Ÿ���� �����Ͽ� �ν��� ����ϴ� ���μ� 0, 1�� ���� ���Եȴ�. Ŀ�ÿ� �Է��� �ƴ� ���� ���嶧 sid�� -1�� �������ִ�.
		//sid�� 1�ΰ��� Ŀ�þ����϶� Ÿ���ʸ� ���� �����ϴ� �ϴ� ����̹Ƿ� Ÿ���� �Է����� �����Ѵ�.
		s->invokeisle->inject(sid == 1 ? s->invokeisle->exttargate : s->invokeisle->extingate, s->invokeisle->exttargate, syncIndice, sid);
	}
	for(Dynagen *dgen = dgenlist;dgen; dgen = dgen->ptrRight) dgen->waitprec();//���޽� �Ҽ� ��� ���̳����� ���� ���� ��� ȣ��
	return dgenlist->ptrLeft->predictv(predwithloss ? ploss : nullx);//��ǥ�� ������ dgen������ ����, �ʿ��ϸ� �� dgen���� Ȯ��
}

Isleloop::Isleloop(Impulse *imp, Dynagen *dgen, sytet head, bytet *name)
{//gate�÷����� �Է� ���, Ÿ�Ե��� �˱����� �����μ� �Ʒ�/�򰡶� �����ͷ� ���� �������� �ʴ´�.
	isleimp = imp;
	isledgen = dgen;
	cntTStep = 0;
	isleRoute = nullx;
	neverTrained = 0;
	strcpy(isleName, name);
	printf("ISLE NAME: %s\n", name);
	isletrc = TRACER(trace(0, name));
	bytet repo[256];
	sprintf(repo, "%s/%s_%s", imp->imppath, imp->impname, dgen->dgenname);
	isletrc->reposet(repo);
	isletrc->migopt(isleimp->impTcr);
	if(isletrc->frozenwgt) fintrain = 1;
	else fintrain = 0;
	nTStep = imp->ntrainStep;
	headisle = head;//0:head�� �ƴ�, 1:�Է°� ���ڵ� head, 2:Ÿ�ٰ� ���ڵ� head, 3:�Է°� Ÿ�� ���� head
	syncurIndice = 0;
	cntoper = cntwait = islebusy= 0;
	sockIndice[0] = sockIndice[1] = 0;
	tadaptisle = 0;
	extingate = exttargate = nullx;
	alignoper = 0;
	lossIndice = dgen->lossIndice;
	finalisle = 0;
	nprotect = 1;//Ŀ�� isle�ܴ̿� �������尡 �Ѱ��̹Ƿ� �ϴ� 1�� ����. 

	szlatent = dgen->szlatent;
	if((headisle == HID || headisle == HAD) && dgen->dgindisc) isindisc = dgen->dgindisc;
	else isindisc = 0;
	if((headisle == HOD || headisle == NHOC || headisle == HAD) && dgen->dgoutdisc) isoutdisc = dgen->dgoutdisc;
	else isoutdisc = 0;
	szembede = dgen->szembede;
	actcode = dgen->actcode;
	rLeaning = dgen->rLeaning;
	fowardit = 0;
	icntTrain = icumTrain = 0;
	sumloss = 0;
	minloss = 10000;
	istableMean = istableMean2 = 0;

	intt rv;
	CRE_MUT_(mutisle, rv);
	if(rv < 0) {
		throwFault(-1, (bytex *)"cre mut fail");
	}
	cydisle = rsc::cydrome;
	cydcijec = cydisle->getcydid();
	CRE_LSEM_(semijec, 0, cydcijec->sem_iden, rv);
	if(rv < 0) {
		throwFault(-1, (bytex *)"cre sem fail\n");
	}
	cydcpeek = cydisle->getcydid();
	CRE_LSEM_(sempeek, 0, cydcpeek->sem_iden, rv);
	if(rv < 0) {
		throwFault(-1, (bytex *)"cre sem fail\n");
	}
	imp->regiisle(this);
}
void Isleloop::decreaseProtect(void) //�����н��̸� ������ �н��Ϸ���� �ʾҴµ� ������ ���� �н��Ϸ��Ͽ�
{							//���̻� �н��� ���� ���ϸ� �ȵǹǷ� ������ �н��Ϸ��� �н��Ϸ�üũ �����ϰ� �Ѵ�.
	LOCK_MUT_(mutisle);
	nprotect--;
	icumTrain = 0;//�����Ͽ� �̶����� MGC�� �����ؼ� �н� ����üũ�� �����ϰ� �Ѵ�.
	UNLOCK_MUT_(mutisle);
}
void Isleloop::finaltrain(Flux *loss)
{
	intt mgc = isletrc->dgenmgc;//�� ����� �� Ŭ���� �� �����ϰ� �н�

	if(fintrain || neverTrained) {//�н��Ϸ��̰ų� final���Ϸ����϶� �н�����ʰ� �����̸�
		return;//�ܺο��� �н� ���Ḧ �����Ѵ�.rTrainBound�� 0�� �����Ǹ� ��� ���Ϸ����� �н����� ����.
	}
	sumloss += loss->at_d(0);
	if(++icntTrain != 3) return;//��.
	floatt avg_loss = sumloss / 3;//3���� ��� ������ ���
	sumloss = 0;
	icntTrain = 0;
	if(avg_loss < minloss - 0.0001) {//�� ����� �������� �� �����ϰ� �н�
		minloss = avg_loss;
		istableMean = 0;
		istableMean2 = 0;
	} else if(istableMean2 > 30 || (istableMean2 > mgc && avg_loss - minloss < 0.005)) {//
		goto LB1;//�ּ���շν� �̰��� 2�� ī��Ʈ�� mgc�� �̻��̸� �̹� ��շν��� �ּ���հ� 
		//��������(���̰� 0.005�̸��̸� ���������� ����, �̰��� ������ �ٽ� �ּҿ� �����ϱⰡ �ð���
		//���� �ɸ���� �н� ���� ������ ���� �ɸ��������Ƿ� ������ ū������ ����) ������ �н� ����.
		//2�� ī��Ʈ�� 30�� ������ �ٽ� �ּҿ� ���� ���ص� �н� ����.
	} else if(isletrc->rTrainBound != 0) {//rTrainBound�� 0�̸� ��� ���Ϸ����� �н��Ϸ������ʰ�
		//��� �н���Ű�ٴٴ� ���̹Ƿ� ī��Ʈ �������� �ʴ´�.
		istableMean++;//�̹� �ν������ ���������� �ּ���շν����� ���� �ּ���շν� ������ �����ʾ����� ī��Ʈ
	}
	printf("%s [%p] _train loss average: %f min: %f step: %d istable: %d istable2: %d count: %d\n", 
		isleName, this, avg_loss, minloss, isleimp->trainstep, istableMean, istableMean2, icumTrain);
	//�� �޼����� �� �н� ���ܸ��� ��µ��� �ʴ´�. �� ��)���� üũ�Ǳ� ������ 
	//if(headisle == HID) {
	//	fintrain = 1; return;
	//} else if(headisle != NHNDI) return;

	//if(headisle != HID && headisle != NHNDI) return;
	//if(headisle != HOD ) return;

	//if(isleType == ENCODE_ISLE) icumTrain++;
	//else
	if(avg_loss < isletrc->rTrainBound) icumTrain++;
	else {
		icumTrain = 0;
		if(istableMean > mgc) {//mgc���̻� ��շν��� �ּҰ� ������ �ȵ�����
			istableMean2++;//2�� ī��Ʈ ����
			istableMean = 0;
		}
	}
	if(icumTrain == 1) {//1�� �����Ͽ� ��տ����� ���ġ �����̸� �н� �Ϸ� �����Ѵ�.
LB1:;	if(isletrc->concurtrain) {//�����н��̸�
			if(nprotect <= 0) {//ilfl.���������� �� �н��Ϸ�Ǵ� �� ������ ���������� �Ϸ�üũ��
				fintrain = 1;// ����ϸ� ���ĺ��� ����������� �Ϸ�üũ �ϰ��Ѵ�.���ڴ��� ���ڴ��� �н������̹Ƿ� ���ڴ� ����.
				if(isleType == ENCODE_ISLE && ((Encodeisle *)this)->decisle) ((Encodeisle *)this)->decisle->decreaseProtect();
				for(Socketisle *s = isleRoute; s; s = s->ptrRight) {//�н� 1�ܰ��̸� ���ڴ�������
					//���� ������Ű�� ���ϰ� �ϰ� �н� 2�ܰ��̸� ���ڴ� �� �н��Ϸ� ���� �����Ҽ��ְ�
					//�Ѵ�. �н� 2�ܰ迡���� ���ڴ� �ܴ̿� ���⿡ ���� �ʴ´�.
					if(isleimp->trainstep < 0 || s->destisle->isleType != DECODE_ISLE) s->destisle->decreaseProtect();
				}
			}
		} else fintrain = 1;
		if(fintrain) {
			istableMean = istableMean2 = icumTrain = 0;//������ ��� 2�ܰ� �н��� �����ϱ�
			minloss = 10000;						//���� �Ϸ�üũ�� �ʱ�ȭ
			isleimp->countDoneTrain();				
			if(isleType == DECODE_ISLE && ((Deocdeisle *)this)->encisle->findectrain == 0) {
				((Deocdeisle *)this)->encisle->findectrain = 1;//���ڴ��� ���ڴ��� ���ڴ� �Ʒ� �Ϸ� ����
				fintrain = 0;//�ٽ� �����Ͽ� 2�ܰ� �н����۵ɼ��ְ� �Ѵ�.
				if(isletrc->concurtrain && ((Deocdeisle *)this)->finalDenseDec == 0) {//�����н��̰� 
					((Deocdeisle *)this)->nprotect = 1;//�������ڴ��� ���ڴ��� �ƴϸ� ���Ӽ��� �����Ͽ�
				}	//2�ܰ� �н� �������� ilfl)�� ����ؾ� �н��Ϸ� �����ɼ��ְ� �ϰ� �������ڴ���  
			}		//���ڴ��� 2�ܰ� �н� ���������� ���Ӱ��谡 ���� ��Ʈ�̹Ƿ� ���Ӽ��� �������� 
		}	//�ʾ� ilfl)�� ������� ����ϰ� �Ͽ� �������ڴ����ڴ��� ���� ���ʷ� ������ ���ڴ����� 
	}		//�н� ����ɼ��ְ� �Ѵ�.
}
void Isleloop::predloop(longx sync_indice)
{
	Flux *r = predislet(islein[0], isleimp->predwithloss ? &predloss : nullx);

	/*if(isleType == DECODE_ISLE) {
		printisle(islein[0], "decoder pred in");
		printisle(islein[1], "decoder pred tar");
		printisle(r, "decoder pred out");
	} else if(isleType == COUPLE_ISLE) {
		printisle(islein[0], "couple pred in");
		printisle(islein[1], "couple pred tar");
		printisle(r, "couple pred out");
	}*/
	eject(r, sync_indice);
	lossout(isleimp->predwithloss ? predloss : nullx, "_pred");
	if(finalisle && isleimp->trainstep == 0) {//���϶��� ����
		r = restoreOrigin(r);//�Է��� 2d�� ��� ����
		//printf("final loss: %p pred: %p\n", predloss, r);
		isledgen->sigprec(r);
	}
}
void Isleloop::lossout(Flux *loss, const bytet *action)
{
	if(finalisle) {
		isledgen->dgenloss[0] = loss;//lossIndice] = loss;//���̳θ� �����ϱ⶧���� �����ʿ����. ���߿� lossIndice���� ����
	}
	if(isletrc->printloss) {
		LOCK_MUT_(isleimp->mutimp);
		printf("%s [%p] %s\n", isleName, this, action);
		if(isleimp->predwithloss) {
			if(loss->fshape[0] > 1) {
				loss->iprinto();//���ڴ��� �ν��� �迭�̹Ƿ� ù��° �͸� ���
			} else loss->printo();
		}
		UNLOCK_MUT_(isleimp->mutimp);
	}
}
void Isleloop::setdgencoup(Dynagen *dgen)
{
	isletrc->convolving = dgen->coupConvolv;
	isletrc->lowbound = dgen->couplowbo;
}
void convert_2d_to_1d(Flux *sor, intt dims[])
{
	if(sor->fdim > 3) {
		dims[0] = sor->fshape[0];
		dims[1] = sor->fshape[1] * sor->fshape[2];//2d�̸� 1d�� ����
		dims[2] = sor->fshape[3];
	} else {
		dims[0] = sor->fshape[0];
		dims[1] = sor->fshape[1];
		dims[2] = sor->fshape[2];
	}
}
void Deocdeisle::encmhead(void)
{
	((Generic *)islenet)->setmhead((Generic *)encisle->islenet);
}
void Coaxisle::setdgendual(void)
{
	isindisc = isledgen->dgindisc;
	isoutdisc = isledgen->dgoutdisc;
}
Impulse *impluse(Tracer *tcr, const bytet *name)
{
	return new(tcr)Impulse(tcr, name);
}
Dynagen *dynagen(sytet t_dynagen, Impulse *imp, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, 
	intt outdiscret, intt embedim, sytet af, floatt lr, const bytet *name)
{
	return new(imp->impTcr)Dynagen(t_dynagen, imp, ingate, targate, latent_sz, indiscret, outdiscret,
		embedim, af, lr, name);
}
Dynagen *dynagen(Impulse *imp, Isleloop *in_zc_isle, Isleloop *tar_zc_isle, bool pred_link,
	sytet af, floatt lr, const bytet *name)
{
	return new(imp->impTcr)Dynagen(imp, in_zc_isle, tar_zc_isle, pred_link, af, lr, name);
}
Dynagen *dynagen(Impulse *imp, Flux *ingate, intt latent_sz, intt indiscret, intt embedim, Isleloop *tar_zc_isle,
	bool pred_link, sytet af, floatt lr, const bytet *name)
{
	return new(imp->impTcr)Dynagen(imp, ingate, latent_sz, indiscret, embedim, tar_zc_isle, pred_link, af, lr, name);
}
Dynagen *dynagen(Impulse *imp, Isleloop *in_zc_isle, Flux *targate, intt latent_sz, intt outdiscret,
	intt embedim, sytet af, floatt lr, const bytet *name)
{
	return new(imp->impTcr)Dynagen(imp, in_zc_isle, targate, latent_sz, outdiscret, embedim, af, lr, name);
}
Dynagen *dynagen(Impulse *imp, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret,
	intt embedim, sytet af, floatt lr, const bytet *name)
{
	return new(imp->impTcr)Dynagen(imp, ingate, targate, latent_sz, indiscret, outdiscret, embedim, af, lr, name);
}