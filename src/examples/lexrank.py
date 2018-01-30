import numpy as np


class LexRank():
    def __init__(self, threshold=0.3, damping_factor=0.1, error_tolerance=0.01):
        self.threshold = threshold
        self.damping_factor = damping_factor
        self.error_tolerance = error_tolerance

    def get_ranking(self, vectors):
        self.n_node = len(vectors)
        self.compute_sim_matrix(vectors)
        self.do_markov_chain()

        score = np.sort(self.p)[::-1]
        ranking = np.argsort(self.p)[::-1]
        return score, ranking

    def do_markov_chain(self, vectors):
        U = np.ones((self.n_node, self.n_node)) / self.n_node
        degrees = np.sum(self.sim_matrix, axis=1, keepdims=True)
        B = self.sim_matrix / degrees
        X = self.damping_factor * U + (1 - self.damping_factor) * B
        X_T = np.transpose(X)

        p_0 = np.ones(self.n_node) / self.n_node
        p_1 = np.dot(X_T, p_0)

        t = 1
        while np.linalg.norm(p_1 - p_0, ord=2) >= self.error_tolerance:
            p_0 = p_1
            p_1 = np.dot(X_T, p_0)
            t += 1

        print("Multiplyed X {} times until convergence.".format(t))
        self.p = p_1

    def compute_sim_matrix(self, vectors):
        self.sim_matrix = np.zeros((self.n_node, self.n_node))
        for i, row in enumerate(self.sim_matrix):
            for j, column in enumerate(row):
                if i <= j:
                    sim = self.get_cosine_sim(vectors[i], vectors[j])
                    if sim >= self.threshold:
                        self.sim_matrix[i, j] = sim
                else:
                    self.sim_matrix[i, j] = self.sim_matrix[j, i]

    def get_cosine_sim(self, x, y):
        inner_product = np.dot(x, y)
        norm_x = np.linalg.norm(x, ord=2)
        norm_y = np.linalg.norm(y, ord=2)
        return inner_product / norm_x / norm_y
