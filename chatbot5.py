

import numpy as np
import mempute as mp
from data import DatasetManager

class KModel:
    def __init__(self, xsz=32, latent_sz=64, embede_dim = 64, nb=10, ep = None, name = None):

        self.dm = DatasetManager('chatbot')
        self.dm.load_vocab()

        self.train_data_iter = self.dm.data_generator(nb, xsz, data_type='train', epoch=ep)#, under_line=8)

        self.train_batch_iter = self.dm.data_generator(nb*2, xsz, data_type='train', epoch=ep)#, under_line=8)

        self.batch_size = nb
        self.logfp = None

        if name is None:
            self.trc = mp.tracer()
        else:
            self.trc = mp.tracer(0, name)
        mp.traceopt(self.trc, 0, 4)
        mp.traceopt(self.trc, 8, 1)
        mp.traceopt(self.trc, 9, 4)
        mp.traceopt(self.trc, 10, 1)
        self.input_data = mp.flux(self.trc, [-1, xsz, 1], mp.variable, mp.tfloat)
        self.target_label = mp.flux(self.trc, [-1, xsz, 1], mp.variable, mp.tfloat)
        self.cell = mp.coaxial(self.input_data, self.target_label, latent_sz, len(self.dm.source_id2word), len(self.dm.target_id2word), embede_dim)

        self.accu_input = mp.flux(self.trc, [-1, xsz, 1], mp.variable, mp.tfloat)
        self.accu_label = mp.flux(self.trc, [-1, xsz, 1], mp.variable, mp.tfloat)
        mp.accuracy(self.cell, self.accu_input, self.accu_label)

    def open_logf(self, mode):
        self.logfp = open(self.model_name, mode)

    def logf(self, format, *args):
        data = format % args
        print(data)
        if self.logfp is not None:
            self.logfp.write(data + '\n')

    def close_logf(self):
        if self.logfp is not None: self.logfp.close()

    def predict(self, input_ids):#입쳑 시퀀스로부터 바로 타겟 시퀀스 예측
        mp.feeda(self.input_data, input_ids)
        self.y_pred = mp.predict(self.cell);
        return mp.eval(self.y_pred)

    def predicts(self, input_ids):
        y_preds = []
        n = input_ids.shape[0]
        i = 0
        while i + self.batch_size <= n:
            in_batch = input_ids[i:i + self.batch_size]
            y_pred = self.predict(in_batch)
            y_preds.append(y_pred)
            i += self.batch_size
        return np.array(y_preds, dtype='i').reshape(-1, y_pred.shape[1], y_pred.shape[2]), i
    
    def accuracy(self, input_ids, target_ids):

        # 테스트용데이터로 rmse오차를 구한다
        predict_res, sz = self.predicts(input_ids)
        target_ids = target_ids[0:sz]
        mp.copya(self.accu_input, predict_res)
        mp.copya(self.accu_label, target_ids)
        error = mp.measure_accuracy(self.cell)
       
        return mp.eval(error), input_ids, predict_res, target_ids

    def recover_sentence(self, sent_ids, id2word):
        #Convert a list of word ids back to a sentence string.
        #for i in sent_ids:
        #    print(i)
        #    print(id2word[i])
        words = list(map(lambda i: id2word[i] if 0 <= i < len(id2word) else '<unk>', sent_ids))

        # Then remove tailing <pad>
        i = len(words) - 1
        while i >= 0 and words[i] == '<pad>':
            i -= 1
        words = words[:i + 1]
        return ' '.join(words)

    def evaluate(self, msg, input_ids, pred_ids, target_ids):
        """Make a prediction and compute BLEU score.
        """
        refs = []
        hypos = []
        
        pred_ids = np.squeeze(pred_ids)
        target_ids = np.squeeze(target_ids)
        
        self.logf("\n=================== %s ===========================", msg)
        if input_ids is not None:
            input_ids = np.squeeze(input_ids)
            for sor, truth, pred in zip(input_ids, target_ids, pred_ids):
                source_sent = self.recover_sentence(sor, self.dm.source_id2word)
                truth_sent = self.recover_sentence(truth, self.dm.target_id2word)
                pred_sent = self.recover_sentence(pred, self.dm.target_id2word)
                self.logf("[Source] %s", source_sent)
                self.logf("[Truth] %s", truth_sent)
                self.logf("[Translated] %s\n", pred_sent)
        else:
            for truth, pred in zip(target_ids, pred_ids):
                truth_sent = self.recover_sentence(truth, self.dm.target_id2word)
                pred_sent = self.recover_sentence(pred, self.dm.target_id2word)
                self.logf("[Truth] %s", truth_sent)
                self.logf("[Translated] %s\n", pred_sent)
        #smoothie = SmoothingFunction().method4
        #bleu_score = corpus_bleu(refs, hypos, smoothing_function=smoothie)
        #return {'bleu_score': bleu_score * 100.}

    def train(self, n_step=None):
        i_step = 0
        while n_step is None or i_step < n_step:
            
            try:
                sample_x, sample_y, ep = next(self.train_data_iter)
                
                #print(sample_x.shape)
                mp.feeda(self.input_data, sample_x)
                mp.feeda(self.target_label, sample_y)
                total_loss = mp.train(self.cell);
                print('loss: ', mp.eval(total_loss)[0])
                
            except StopIteration:
                break
            
            if i_step % 20 == 0:
               
                sample_x, sample_y, _ = next(self.train_batch_iter)
                train_error, train_input, train_predict, train_right = self.accuracy(sample_x, sample_y)
                self.evaluate('train batch', train_input, train_predict, train_right)
                self.logf("epoch: %d step: %d, train(A): %f, test(B): %f, B-A: %f",
                    ep, i_step, train_error, test_error, test_error-train_error)
                print("step #: ", i_step)

            #if i_step % 826 == 0:
            #    mp.save_weight(self.trc)
            i_step += 1
        #mp.save_weight(self.trc)

kmodel = KModel(name='chatt5')
kmodel.train(80000)