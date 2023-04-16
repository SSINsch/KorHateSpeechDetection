import sklearn.metrics as metrics


class KorHateTrainer:
    def __init__(self, model, optimizer, criterion, train_iter, val_iter, device):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.device = device
        self.criterion = criterion

    def train(self):
        self.model.train()

        corrects, total_loss = 0, 0
        for b, batch in enumerate(self.train_iter):
            print('x')
            # comments 는 x, hate(label)은 y로 두고
            x, y = (batch.comments.to(self.device), batch.comments_char.to(self.device)), batch.hate.to(self.device)
            # gradient 0으로 세팅해두고
            self.optimizer.zero_grad()
            # model 돌리고
            prediction = self.model(x)
            # loss 구해서 backprop
            loss = self.criterion(prediction, y)
            total_loss = total_loss + loss.item()
            loss.backward()
            self.optimizer.step()

        size = len(self.train_iter.dataset)
        avg_loss = total_loss / size

        return avg_loss

    def evaluate(self):
        self.model.eval()

        corrects, total_loss = 0, 0

        for b, batch in enumerate(self.val_iter):
            x, y = batch.comments.to(self.device), batch.hate.to(self.device)
            prediction = self.model(x)
            loss = self.criterion(prediction, y)
            total_loss = total_loss + loss.item()
            corrects = corrects + (prediction.max(1)[1].view(y.size()).data == y.data).sum()

            y = y.data
            y = y.to("cpu")
            y = y.detach().numpy()
            p = prediction.max(1)[1].data
            p = p.to("cpu")
            p = p.detach().numpy()

        print('**** Look only last batch case ****')
        print(metrics.classification_report(y, p))

        size = len(self.val_iter.dataset)
        avg_loss = total_loss / size
        avg_accuracy = 100.0 * corrects / size

        return avg_loss, avg_accuracy
