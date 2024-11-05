
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

Flux *Impulse::train(intt *n_train) //오차값 확인은 임펄스 소속 다이나젠의 dgenloss를 참조한다.
{
	Flux *tloss;
	if(tadapt < 2) {
		if(tadapt == 0) coupleisleThr();
		dgenlist->ptrLeft->dumyloss();//초기에 로스가 적재되지 않아 밑에서 tloss가 널로 설정되는 것을 방지
		if(impTcr->concurtrain) {//동시학습이면 최하위선행은 선행이 없으므로 학습완료체크 허용을 해줄 isle이
			for(SyncInvoke *s = synchor;s; s = s->ptrRight) {//없으므로 여기서 초기해 해준다.
				s->invokeisle->decreaseProtect();
			}
		}
		tadapt = 2;
	}
	if(runPredict) {//중간 인코더isle의 학습이 완료되지 않은 중간에 예측을 수행하고 다시 학습을
		runPredict = 0;//수행하는 경우 예측과정에서 목표값도 주이질 경우 커플isle의 오퍼랜드가  
		retrain();//정렬되어 이후부터는 ieij)에서 커플isle의 입력과 타겟 양측이 주입되지않으면
				//블럭이 걸리는데 인코더isle의 학습이 끝나지 않았으므로 커플isle이 입력이 주입되지
				//않으므로 전체 블럭이 걸린다. 따라서 이경우 다시 리셋한다. 학습중간에 예측을 수행
	}			//하는 경우 학습때 주입된 이전 데이턷들은 ieie)에서 제거된다.
	predwithloss = 1;
	trainstep = init_trainstep;
	if(impTcr->concurtrain > 1) {//전체 동시학습(단계별 학습완료후 포워드가 아닌) 2단계이면 
		finalSync = 1;				//매 스텝 전체 isle 학습 수행 후 리턴
		itrainisle = (trainstep == 1 ? 0 : idonetrain);//2단계 학습이면 iril)에서
	} //2단계 학습 완료되지 않는 디코더만 학습 카운트 되므로 초기값으로 학습 완료된 isle갯수를 설정한다.
	for(SyncInvoke *s = synchor;s; s = s->ptrRight) {//sockid가 1인경우는 커플아일일때 
		//타겟쪽만 직접 주입하는 하는 경우이므로 타겟을 입력으로 주입한다.
		s->invokeisle->inject(s->sockid == 1 ? s->invokeisle->exttargate : s->invokeisle->extingate, s->invokeisle->exttargate, syncIndice, s->sockid);
	}
	syncIndice++;
	if(finalSync) waittraining();//inject에의해 플로우 컨트롤 되므로 대기 필요없으나(실행중이면
	//위에서 다음 스텝 주입할때 대기된다) 동시학습 2단계이면 매 스텝 전isle이 완료될때까지 대기
	dgenlist->ptrLeft->predictv(&tloss);//대표로 마지막 dgen것으로 리턴, 필요하면 각 dgen별록 확인
	if(init_trainstep != -1) {//2단계 학습이 시작되지 않았으면(1단계 학습 수행중이면)
		if(ntrainisle == idonetrain) {//1단계 학습완료 시점에 2단계 학습 설정.
			init_trainstep = -1;//2단계 학습 표시(이후부터는 디코더만 학습 수행됨)
			idonetrain -= ndecisle;//디코더 isle객수를 빼주어 2단계 학습은 디코더 개수만 카운되면 전체 학습 완료체크되게 한다.
		}
	}
	tloss->printo();
	if(n_train) *n_train = ntrainisle - idonetrain;
	return tloss;//최종 isle의 손실만 적재되므로 최종isle에 도달하기전까진 기본값인 0가 출력된다.
}
Flux *Impulse::predict(Flux **ploss) //예측값의 확인은 임펄스 소속 다이나젠의 dgenpred를 참조한다.
{
	sytet sid;
	if(tadapt == 0) {
		coupleisleThr();
		tadapt = 1;
	}
	//학습중간에 예측을 수행하는 경우 학습때 주입된 이전 데이턷들은 ieie)에서 제거된다.
	if(ploss) predwithloss = 1;
	else predwithloss = 0;
	runPredict = 1;
	trainstep = 0;
	finalSync = 0;
	syncIndice++;
	for(SyncInvoke *s = synchor;s; s = s->ptrRight) {
		sid = s->sockid;
		if(predwithloss == 0) {//평가때 로스 계산 없는 경우.
			if(sid == 1) continue;//타겟 실행이 아니면 타겟쪽 주입은 하지 않는다.
			//else if(s->sockid == -2) sid = -1;//양측 동시 주입이면 입력측만 주입하게 한다.
			sid = -1;//소켓아이디가 0이던 -2(양측 동시 주입)이던 -1로하여 ijed)에서 입력만 주입되게 한다.
		}//else 평가때 타겟을 주입하여 로스도 계산하는 경우로서 0, 1이 따로 주입된다. 커플에 입력이 아닌 경우는 빌드때 sid는 -1로 설정되있다.
		//sid가 1인경우는 커플아일일때 타겟쪽만 직접 주입하는 하는 경우이므로 타겟을 입력으로 주입한다.
		s->invokeisle->inject(sid == 1 ? s->invokeisle->exttargate : s->invokeisle->extingate, s->invokeisle->exttargate, syncIndice, sid);
	}
	for(Dynagen *dgen = dgenlist;dgen; dgen = dgen->ptrRight) dgen->waitprec();//임펄스 소속 모든 다이나젠의 예측 종료 대기 호출
	return dgenlist->ptrLeft->predictv(predwithloss ? ploss : nullx);//대표로 마지막 dgen것으로 리턴, 필요하면 각 dgen별록 확인
}

Isleloop::Isleloop(Impulse *imp, Dynagen *dgen, sytet head, bytet *name)
{//gate플럭스는 입력 모양, 타입등을 알기위한 정보로서 훈련/평가때 포인터로 직접 참조되지 않는다.
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
	headisle = head;//0:head가 아님, 1:입력값 인코딩 head, 2:타겟값 인코딩 head, 3:입력과 타겟 연결 head
	syncurIndice = 0;
	cntoper = cntwait = islebusy= 0;
	sockIndice[0] = sockIndice[1] = 0;
	tadaptisle = 0;
	extingate = exttargate = nullx;
	alignoper = 0;
	lossIndice = dgen->lossIndice;
	finalisle = 0;
	nprotect = 1;//커플 isle이외는 오러랜드가 한개이므로 일단 1개 설정. 

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
void Isleloop::decreaseProtect(void) //동시학습이면 선행이 학습완료되지 않았는데 후행이 먼저 학습완료하여
{							//더이상 학습을 수행 안하면 안되므로 선행이 학습완료후 학습완료체크 수행하게 한다.
	LOCK_MUT_(mutisle);
	nprotect--;
	icumTrain = 0;//리셋하여 이때부터 MGC번 연속해서 학습 오차체크를 수행하게 한다.
	UNLOCK_MUT_(mutisle);
}
void Isleloop::finaltrain(Flux *loss)
{
	intt mgc = isletrc->dgenmgc;//이 계수가 더 클수록 더 정밀하게 학습

	if(fintrain || neverTrained) {//학습완료이거나 final아일루프일때 학습종료않게 설정이면
		return;//외부에서 학습 종료를 결정한다.rTrainBound가 0로 설정되면 모든 아일루프가 학습종료 없다.
	}
	sumloss += loss->at_d(0);
	if(++icntTrain != 3) return;//ㄱ.
	floatt avg_loss = sumloss / 3;//3번의 평균 오차를 계산
	sumloss = 0;
	icntTrain = 0;
	if(avg_loss < minloss - 0.0001) {//이 계수가 작을수록 더 정밀하게 학습
		minloss = avg_loss;
		istableMean = 0;
		istableMean2 = 0;
	} else if(istableMean2 > 30 || (istableMean2 > mgc && avg_loss - minloss < 0.005)) {//
		goto LB1;//최소평균로스 미갱신 2차 카운트가 mgc번 이상이면 이번 평균로스가 최소평균과 
		//같아지는(차이가 0.005미만이면 같은것으로 간주, 이값이 작으면 다시 최소에 도달하기가 시간이
		//오래 걸릴경우 학습 종료 설정이 오래 걸릴수있으므로 적당히 큰값으로 설정) 시점에 학습 종료.
		//2차 카운트가 30이 넘으면 다시 최소에 도달 못해도 학습 종료.
	} else if(isletrc->rTrainBound != 0) {//rTrainBound이 0이면 모든 아일루프를 학습완료하지않고
		//계속 학습시키겟다는 것이므로 카운트 증가하지 않는다.
		istableMean++;//이번 로스평균이 이제까지의 최소평균로스보다 적어 최소평균로스 갱신이 되지않았으면 카운트
	}
	printf("%s [%p] _train loss average: %f min: %f step: %d istable: %d istable2: %d count: %d\n", 
		isleName, this, avg_loss, minloss, isleimp->trainstep, istableMean, istableMean2, icumTrain);
	//위 메세지는 메 학습 스텝마다 출력되지 않는다. 위 ㄱ)에서 체크되기 때문에 
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
		if(istableMean > mgc) {//mgc번이상 평균로스의 최소값 갱신이 안됐으면
			istableMean2++;//2차 카운트 증가
			istableMean = 0;
		}
	}
	if(icumTrain == 1) {//1번 연속하여 평균오차가 허용치 이하이면 학습 완료 설정한다.
LB1:;	if(isletrc->concurtrain) {//동시학습이면
			if(nprotect <= 0) {//ilfl.하위선행이 현 학습완료되는 현 시점에 상위후행의 완료체크를
				fintrain = 1;// 허락하면 이후부터 상위후행들은 완료체크 하게한다.디코더도 인코더의 학습후행이므로 디코더 설정.
				if(isleType == ENCODE_ISLE && ((Encodeisle *)this)->decisle) ((Encodeisle *)this)->decisle->decreaseProtect();
				for(Socketisle *s = isleRoute; s; s = s->ptrRight) {//학습 1단계이면 디코더간에는
					//방지 헤제시키지 못하게 하고 학습 2단계이면 디코더 간 학습완료 방지 해제할수있게
					//한다. 학습 2단계에서는 디코더 이외는 여기에 오지 않는다.
					if(isleimp->trainstep < 0 || s->destisle->isleType != DECODE_ISLE) s->destisle->decreaseProtect();
				}
			}
		} else fintrain = 1;
		if(fintrain) {
			istableMean = istableMean2 = icumTrain = 0;//디코일 경우 2단계 학습을 시작하기
			minloss = 10000;						//위해 완료체크를 초기화
			isleimp->countDoneTrain();				
			if(isleType == DECODE_ISLE && ((Deocdeisle *)this)->encisle->findectrain == 0) {
				((Deocdeisle *)this)->encisle->findectrain = 1;//디코더의 인코더에 디코더 훈련 완료 설정
				fintrain = 0;//다시 리셋하여 2단계 학습시작될수있게 한다.
				if(isletrc->concurtrain && ((Deocdeisle *)this)->finalDenseDec == 0) {//동시학습이고 
					((Deocdeisle *)this)->nprotect = 1;//최종인코더의 디코더가 아니면 종속성을 복원하여
				}	//2단계 학습 과정에서 ilfl)를 통과해야 학습완료 설정될수있게 하고 최종인코더의  
			}		//디코더는 2단계 학습 과정에서는 종속관계가 없는 루트이므로 종속성을 설정하지 
		}	//않아 ilfl)를 제약없이 통과하게 하여 최종인코더디코더로 부터 차례로 오리진 디코더까지 
	}		//학습 뫈료될수있게 한다.
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
	if(finalisle && isleimp->trainstep == 0) {//평가일때만 수행
		r = restoreOrigin(r);//입력이 2d일 경우 복사
		//printf("final loss: %p pred: %p\n", predloss, r);
		isledgen->sigprec(r);
	}
}
void Isleloop::lossout(Flux *loss, const bytet *action)
{
	if(finalisle) {
		isledgen->dgenloss[0] = loss;//lossIndice] = loss;//파이널만 적재하기때문에 구분필요없다. 나중에 lossIndice관련 제거
	}
	if(isletrc->printloss) {
		LOCK_MUT_(isleimp->mutimp);
		printf("%s [%p] %s\n", isleName, this, action);
		if(isleimp->predwithloss) {
			if(loss->fshape[0] > 1) {
				loss->iprinto();//인코더의 로스는 배열이므로 첫번째 것만 출력
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
		dims[1] = sor->fshape[1] * sor->fshape[2];//2d이면 1d로 변경
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