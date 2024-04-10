from typing import List
import numpy as np
import wandb

class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0

        for q in range(len(predicted)):
            in_count = 0
            for doc in predicted[q]:
                if doc in actual[q]:
                    in_count += 1
            precision += in_count / len(predicted[q])

        precision /= len(predicted)

        return precision
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0

        for q in range(len(predicted)):
            in_count = 0
            for doc in actual[q]:
                if doc in predicted[q]:
                    in_count += 1
            recall += in_count / len(actual[q])

        recall /= len(predicted)

        return recall
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        f1 = 0.0

        for q in range(len(predicted)):
            tp = 0
            for doc in actual[q]:
                if doc in predicted[q]:
                    tp += 1
            f1 += (2 * tp) / (len(predicted[q]) + len(actual[q]))

        f1 /= len(predicted)

        return f1
    
    def calculate_AP(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0

        tp = 0
        cnt = 0
        for doc in predicted:
            cnt += 1
            if doc in actual:
                tp += 1
            AP += tp / cnt

        AP /= len(predicted)

        return AP
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = 0.0

        for q in range(len(predicted)):
            MAP += self.calculate_AP(actual[q], predicted[q])

        MAP /= len(predicted)

        return MAP
    
    def calculate_DCG(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0

        for idx, doc in enumerate(predicted):
            if doc in actual:
                if idx == 0:
                    DCG += len(actual) - actual.index(doc)
                else:
                    DCG += (len(actual) - actual.index(doc)) / np.log2(idx + 1)

        return DCG
    
    def calculate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0

        for q in range(len(predicted)):
            NDCG += self.calculate_DCG(actual[q], predicted[q]) * self.calculate_DCG(actual[q], actual[q])

        return NDCG
    
    def calculate_RR(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RR = 0.0

        for idx, doc in enumerate(predicted):
            if doc in actual:
                RR = 1 / (idx + 1)
                break

        return RR
    
    def calculate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = 0.0

        for q in range(len(predicted)):
            MRR += self.calculate_RR(actual[q], predicted[q])

        MRR /= len(predicted)

        return MRR

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """

        print(f"name = {self.name}")
        print(f"precision = {precision}")
        print(f"recall = {recall}")
        print(f"f1 = {f1}")
        print(f"map = {map}")
        print(f"ndcg = {ndcg}")
        print(f"mrr = {mrr}")      

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        
        wandb.init(project="mir-project")
        wandb.log({
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Mean Average Precision": map,
            "Normalized Discounted Cumulative Gain": ndcg,
            "Mean Reciprocal Rank": mrr
        })

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.calculate_DCG(actual, predicted)
        ndcg = self.calculate_NDCG(actual, predicted)
        rr = self.calculate_RR(actual, predicted)
        mrr = self.calculate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)

if __name__ == '__main__':
    eval = Evaluation('test')
    eval.calculate_evaluation([['tt15239678', 'tt1160419', 'tt0087182', 'tt0142032', 'tt31378509'], # query: Dune
                               ['tt9362722', 'tt16360004', 'tt4633694', 'tt0145487', 'tt10872600'], # query: spider-man spider-verse
                               ['tt0133093', 'tt10838180', 'tt0234215', 'tt0242653', 'tt0106062']], # query: matrix
                               [['tt0142032', 'tt1160419', 'tt0087182', 'tt1910516', 'tt10466872'],
                                ['tt4633694', 'tt9362722', 'tt1872181', 'tt0316654', 'tt2250912'],
                                ['tt0088944', 'tt10838180', 'tt0410519', 'tt0234215', 'tt30749809']])
    