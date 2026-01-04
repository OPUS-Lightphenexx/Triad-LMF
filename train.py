import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from module.models import TwoFeatureLMF, ThreeFeatureLMF, AllFeatureModel
from utils.data_utils import prepare_data, print_and_save_dataset_info,set_random_seed
from utils.visualization import visualize_results_with_test_loss


def _macro_metrics_from_preds(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    """
    计算多分类 Macro-Precision / Macro-Recall / Macro-F1。
    y_true/y_pred: (N,) 的整型标签张量
    """
    y_true = y_true.view(-1).to(torch.long).cpu()
    y_pred = y_pred.view(-1).to(torch.long).cpu()
    C = int(num_classes)

    cm = torch.bincount(y_true * C + y_pred, minlength=C*C).reshape(C, C).to(torch.float32)
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    eps = 1e-12
    prec_c = tp / (tp + fp + eps)
    rec_c  = tp / (tp + fn + eps)
    f1_c   = 2 * prec_c * rec_c / (prec_c + rec_c + eps)

    return {
        "precision_macro": prec_c.mean().item(),
        "recall_macro": rec_c.mean().item(),
        "f1_macro": f1_c.mean().item(),
    }


def train_and_validate_model(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train = [x.to(device) for x in X_train]
    y_train = y_train.to(device).long()
    X_test = [x.to(device) for x in X_test]
    y_test = y_test.to(device).long()

    train_losses, test_losses, test_accuracies, train_accuracies = [], [], [], []

    train_samples = X_train[0].shape[0]
    test_samples = X_test[0].shape[0]

    # 推断类别数
    num_classes = int(torch.max(torch.cat([y_train, y_test], dim=0)).item() + 1)

    # 记录四个指标的“最高值”
    best_overall_acc = {"value": 0.0, "epoch": 0}        # Overall Accuracy（与你每轮的 Test Acc 相同的定义）
    best_macro_prec  = {"value": 0.0, "epoch": 0}
    best_macro_rec   = {"value": 0.0, "epoch": 0}
    best_macro_f1    = {"value": 0.0, "epoch": 0}

    for e in range(epochs):
        model.train()
        sum_train_loss = 0.0
        sum_train_samples = 0
        train_correct = 0
        train_total = 0

        perm = torch.randperm(train_samples, device=device)

        for i in range(0, train_samples, batch_size):
            end_idx = min(i + batch_size, train_samples)
            idx = perm[i:end_idx]
            batch_X1 = X_train[0][idx]
            batch_X2 = X_train[1][idx]
            batch_X3 = X_train[2][idx]
            batch_y  = y_train[idx]

            optimizer.zero_grad()
            output = model(batch_X1, batch_X2, batch_X3)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            bs = batch_y.size(0)
            sum_train_loss += loss.item() * bs
            sum_train_samples += bs

            pred = output.argmax(dim=1)
            train_correct += (pred == batch_y).sum().item()
            train_total += bs

        avg_train_loss = sum_train_loss / sum_train_samples
        train_losses.append(avg_train_loss)
        train_acc = train_correct / train_total
        train_accuracies.append(train_acc)

        # Eval
        model.eval()
        sum_test_loss = 0.0
        correct = 0
        total = 0
        y_true_all, y_pred_all = [], []
        with torch.inference_mode():
            for i in range(0, test_samples, batch_size):
                end_idx = min(i + batch_size, test_samples)
                batch_X1 = X_test[0][i:end_idx]
                batch_X2 = X_test[1][i:end_idx]
                batch_X3 = X_test[2][i:end_idx]
                batch_y  = y_test[i:end_idx]

                test_output = model(batch_X1, batch_X2, batch_X3)
                loss = criterion(test_output, batch_y)
                bs = batch_y.size(0)
                sum_test_loss += loss.item() * bs
                pred = test_output.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += bs

                y_true_all.append(batch_y)
                y_pred_all.append(pred)

        test_loss = sum_test_loss / total
        test_losses.append(test_loss)
        accuracy = correct / total
        test_accuracies.append(accuracy)


        y_true_all = torch.cat(y_true_all, dim=0)
        y_pred_all = torch.cat(y_pred_all, dim=0)
        macro = _macro_metrics_from_preds(y_true_all, y_pred_all, num_classes)


        if accuracy > best_overall_acc["value"]:
            best_overall_acc = {"value": float(accuracy), "epoch": e + 1}
        if macro["precision_macro"] > best_macro_prec["value"]:
            best_macro_prec = {"value": float(macro["precision_macro"]), "epoch": e + 1}
        if macro["recall_macro"] > best_macro_rec["value"]:
            best_macro_rec = {"value": float(macro["recall_macro"]), "epoch": e + 1}
        if macro["f1_macro"] > best_macro_f1["value"]:
            best_macro_f1 = {"value": float(macro["f1_macro"]), "epoch": e + 1}

        print(f"Epoch {e+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {accuracy:.4f} | "
              f"Macro P/R/F1: {macro['precision_macro']:.4f}/{macro['recall_macro']:.4f}/{macro['f1_macro']:.4f}")


    print(
        f"Best (on test): "
        f"Acc {best_overall_acc['value']:.4f} @ epoch {best_overall_acc['epoch']} | "
        f"Precision {best_macro_prec['value']:.4f} @ epoch {best_macro_prec['epoch']} | "
        f"Recall{best_macro_rec['value']:.4f} @ epoch {best_macro_rec['epoch']} | "
        f"F1 {best_macro_f1['value']:.4f} @ epoch {best_macro_f1['epoch']}"
    )

    return train_losses, test_losses, test_accuracies, train_accuracies


def train(data_dir, num_epochs):
    """主训练逻辑"""
    set_random_seed(seed=48)
    try:
        X1, X2, X3, targets, num_classes = prepare_data(data_dir)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(
        X1, X2, X3, targets, test_size=0.3, random_state=18
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    X1_train, X2_train, X3_train, y_train = [data.to(device) for data in (X1_train, X2_train, X3_train, y_train)]
    X1_test, X2_test, X3_test, y_test = [data.to(device) for data in (X1_test, X2_test, X3_test, y_test)]

    print_and_save_dataset_info(X1_train, X2_train, X3_train, y_train, "Training")
    print_and_save_dataset_info(X1_test, X2_test, X3_test, y_test, "Testing")

    input_dims_feat12 = (X1.shape[1], X2.shape[1])
    input_dims_feat23 = (X2.shape[1], X3.shape[1])
    input_dims_feat13 = (X1.shape[1], X3.shape[1])

    hidden_dims = (256, 256)
    dropout_rates = (0.5, 0.5, 0.5)
    rank_1=100
    output_dim_1 = 100

    all_input_dims = (output_dim_1, output_dim_1, output_dim_1)
    hidden_dims_three = (36, 36, 36)
    dropout_three = (0.5, 0.5, 0.5, 0.5)
    output_dim = num_classes
    rank_2 = 8

    model_feat12 = TwoFeatureLMF(input_dims=input_dims_feat12, hidden_dims=hidden_dims,
                                 dropout=dropout_rates, output_dim=output_dim_1, rank=rank_1)
    model_feat23 = TwoFeatureLMF(input_dims=input_dims_feat23, hidden_dims=hidden_dims,
                                 dropout=dropout_rates, output_dim=output_dim_1, rank=rank_1)
    model_feat13 = TwoFeatureLMF(input_dims=input_dims_feat13, hidden_dims=hidden_dims,
                                 dropout=dropout_rates, output_dim=output_dim_1, rank=rank_1)
    three_LMF = ThreeFeatureLMF(all_input_dims, hidden_dims_three, dropout_three,
                                output_dim=output_dim, rank=rank_2)

    model = AllFeatureModel(model_feat12, model_feat23, model_feat13, three_LMF)

    with torch.no_grad():
        num_classes_int = int(num_classes)
        counts = torch.bincount(y_train.to(torch.long), minlength=num_classes_int).float()
        class_w = (counts.max() / (counts + 1e-8)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    print(f"model_feat12 输入维度: {input_dims_feat12}")
    print(f"model_feat23 输入维度: {input_dims_feat23}")
    print(f"model_feat13 输入维度: {input_dims_feat13}")
    print(f"three_LMF 输入维度: {all_input_dims}")
    print(f"最终输出维度（类别数量）: {output_dim}")

    X_train = (X1_train, X2_train, X3_train)
    X_test = (X1_test, X2_test, X3_test)

    train_losses, test_losses, test_accuracies, train_accuracies = train_and_validate_model(
        model, criterion, optimizer, X_train, y_train, X_test, y_test, num_epochs
    )

    visualize_results_with_test_loss(
        train_losses=train_losses,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        train_accuracies=train_accuracies,
        save_folder="D:\\Desktop\\plott",
        filename="training_results1",
        format='png',
        dpi=600
    )

