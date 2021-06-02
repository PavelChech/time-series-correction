import matplotlib.pyplot as plt


class Graphics:

    @staticmethod
    def linear_graph(title, data):
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        data.plot(title=title, kind='line', x=list(data)[0], y=list(data)[1], ax=ax)
        plt.show()

    # @staticmethod
    # def histogram(data):
    #     plt.bar(data[], counts)
    #     plt.figure(figsize=(12, 8))
    #     ax = plt.gca()
    #     data.hist(title=title, kind='line', x=list(data)[0], y=list(data)[1], ax=ax)
    #     plt.show()
