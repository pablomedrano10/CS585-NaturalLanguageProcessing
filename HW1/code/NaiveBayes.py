import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        #TODO: Initalize parameters
        self.Train(data.X,data.Y)

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()
        
        #The lenght of vocab is equal to the number of columns of the matrix X
        self.vocab_len = X.shape[1]
        
        #Frequency of words appearing in positive reviews
        self.count_positive = X[np.where(Y == 1)[0], :].sum(axis = 0)

        #Frequency of words appearing in negative reviews
        self.count_negative = X[np.where(Y == -1)[0], :].sum(axis = 0)

        #Total count of positive reviews labelled as 1 in the matrix Y
        self.num_positive_reviews = float((Y[Y == 1]).shape[0])

        #Total count of negative reviews labelled as -1 in the matrix Y
        self.num_negative_reviews = float((Y[Y == -1]).shape[0])

        #Total number of words that appeat in positive reviews
        self.total_positive_words = (X[np.where(Y == 1)[0], :].sum(axis = 0)).sum()

        #Total number of words that appear in negative reviews
        self.total_negative_words = (X[np.where(Y == -1)[0], :].sum(axis = 0)).sum()

        #P(+)
        self.P_positive = self.total_positive_words/(self.total_positive_words+self.total_negative_words)

        #P(-)
        self.P_negative = self.total_negative_words/(self.total_positive_words+self.total_negative_words)

        #N+alfa*d
        self.deno_pos = self.total_positive_words+self.ALPHA*self.vocab_len
        self.deno_neg = self.total_negative_words+self.ALPHA*self.vocab_len
        
        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X):
        #TODO: Implement Naive Bayes Classification
        pred_labels = []
    
        #In this for loop the log pos and neg log prob for each review is computed from its words.
        #A label 1 is given if the pos log prob is bigger then the neg log prob else a label -1 is given to that review        
        sh = X.shape[0]
        for i in range(sh):
            z = X[i].nonzero()
            pos_prob = log(self.P_positive)
            neg_prob = log(self.P_negative)
            for j in range(len(z[0])):
                # Look at each feature
                pos_prob += log((self.count_positive[0, z[1][j]]+self.ALPHA)/(self.total_positive_words+self.ALPHA*self.vocab_len))
                neg_prob += log((self.count_negative[0, z[1][j]]+self.ALPHA)/(self.total_negative_words+self.ALPHA*self.vocab_len))
            
            if (pos_prob > neg_prob):            # Predict positive
                pred_labels.append(1.0)
            else:                                # Predict negative
                pred_labels.append(-1.0)
                
        return pred_labels

    def LogSum(self, logx, logy):   
        # TO DO: Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))

    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
            
    def PredictProb(self, test, indexes):
        
            #In this for loop the log-sum-exp trick is done to compute the pos and neg prob of the first 10 reviews
            #in the test data. As before a label 1 is given if the pos prob is bigger than the neg prob else the label
            # given is -1
            for i in indexes:
                # TO DO: Predict the probability of the i_th review in test being positive review
                # TO DO: Use the LogSum function to avoid underflow/overflow
                predicted_label = 0
                                
                z = test.X[i].nonzero()
                pos_prob = log(self.P_positive)
                neg_prob = log(self.P_negative)
                for j in range(len(z[0])):
                    # Look at each feature
                    pos_prob += log((self.count_positive[0, z[1][j]]+self.ALPHA)/(self.total_positive_words+self.ALPHA*self.vocab_len))
                    neg_prob += log((self.count_negative[0, z[1][j]]+self.ALPHA)/(self.total_negative_words+self.ALPHA*self.vocab_len))
                
                den = self.LogSum(pos_prob, neg_prob)
                    
                predicted_prob_positive = exp(pos_prob-den)
                predicted_prob_negative = exp(neg_prob-den) 
                
                if predicted_prob_positive > predicted_prob_negative:
                    predicted_label = 1.0
                else:
                    predicted_label = -1.0
                
                #print test.Y[i], test.X_reviews[i]
                # TO DO: Comment the line above, and uncomment the line below
                print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i])
        
    

    def PredictThres(self, test, thresholds):
        
        precision = []
        recall = []
        
        #in this for loop the log-sum-exp trick is done for each document as in the function before but now the label 1
        #is bigger than a threshold and -1 if it is not. Then this predicted label is compared to the real one
        #and it is determined if the prediction was a true positive, a false positive, a true negative or a false negative
        #With all these values a confusion matrix is built to pass it to the function EvalPrecision and EvalRecall that
        #return the precision and recall for each threshold
        for threshold in thresholds:
            confMatrix = []
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for i in range(test.X.shape[0]):
                # TO DO: Predict the probability of the i_th review in test being positive review
                # TO DO: Use the LogSum function to avoid underflow/overflow
                predicted_label = 0
                                
                z = test.X[i].nonzero()
                pos_prob = log(self.P_positive)
                neg_prob = log(self.P_negative)
                for j in range(len(z[0])):
                    # Look at each feature
                    pos_prob += log((self.count_positive[0, z[1][j]]+self.ALPHA)/(self.total_positive_words+self.ALPHA*self.vocab_len))
                    neg_prob += log((self.count_negative[0, z[1][j]]+self.ALPHA)/(self.total_negative_words+self.ALPHA*self.vocab_len))
                
                den = self.LogSum(pos_prob, neg_prob)
                    
                predicted_prob_positive = exp(pos_prob-den)
                predicted_prob_negative = exp(neg_prob-den) 
                
                if predicted_prob_positive > threshold:
                    predicted_label = 1.0
                else:
                    predicted_label = -1.0
                    
                if predicted_label == 1.0:
                    if test.Y[i] == 1.0:
                        tp += 1
                    else:
                        fp += 1
                elif predicted_label == -1.0:
                    if test.Y[i] == -1.0:
                        tn += 1
                    else:
                        fn += 1
                        
            confMatrix.append(tp)
            confMatrix.append(fp)
            confMatrix.append(tn)
            confMatrix.append(fn)
            print("threshold: ", threshold)
            print("True positive: ", tp)
            print("False positive: ", fp)
            print("False negative: ", fn)
            print("True negative: ", tn)
            print(self.EvalPrecision(confMatrix))
            print(self.EvalRecall(confMatrix))
            precision.append(self.EvalPrecision(confMatrix))
            recall.append(self.EvalRecall(confMatrix))
            
        plt.figure()
        plt.plot(precision, recall, 'ro')
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.show()
        
        return
                    
    #This function has a confusion matrix as a parameter and returns the precision for it
    def EvalPrecision(self, confMatrix):
        return confMatrix[0]/(confMatrix[0] + confMatrix[1])
    
    #This function has a confusion matrix as a parameter and returns the recall for it
    def EvalRecall(self, confMatrix):
        return confMatrix[0]/(confMatrix[0] + confMatrix[3])

    # Evaluate performance on test data 
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()
    
    #In this function the weight for each word is computed using the formula another student posted on piazza
    #Then the top 20 positive and negative are printed
    def Features(self, data):
        features_pos = []
        features_neg = []
        
        for i in range(0, self.vocab_len):
            
            pos_prob = log((self.count_positive[0, i]+self.ALPHA)/(self.total_positive_words+self.ALPHA*self.vocab_len))
            neg_prob = log((self.count_negative[0, i]+self.ALPHA)/(self.total_negative_words+self.ALPHA*self.vocab_len))
            
            pos_word_weight = ((pos_prob - neg_prob)*self.count_positive[0, i] - self.count_negative[0, i])
            neg_word_weight = ((neg_prob - pos_prob)*self.count_negative[0, i] - self.count_positive[0, i])
            
            features_pos.append((data.vocab.GetWord(i), pos_word_weight))
            features_neg.append((data.vocab.GetWord(i), neg_word_weight))
        
        features_pos.sort(key = lambda x: -x[1])
        features_neg.sort(key = lambda x: -x[1])
        
        print(features_pos[:20])
        print(features_neg[:20])


if __name__ == "__main__":
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Evaluating")
    print("Test Accuracy: ", nb.Eval(testdata))
    nb.PredictProb(testdata, range(10))
    nb.PredictThres(testdata, np.arange(0, 1, 0.1))
    nb.Features(traindata)

