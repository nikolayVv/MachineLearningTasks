import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(p, y, normalize = True):

    c = np.zeros((y.shape[-1], y.shape[-1]), dtype = np.float32)

    for i in range(c.shape[0]):

        class_idx = y[:, i] != 0.0

        class_p = p[class_idx]
        class_y = y[class_idx]

        labels_p = np.argmax(class_p, axis = -1)

        np.add.at(c[:, i], labels_p, 1.0)

    return c if not normalize else (c/c.sum(axis = 0)[None, ...], c.sum(axis = 0)[None, ...])

def get_precission(c):

    p = np.zeros(c.shape[1])

    for i in range(c.shape[1]):

        tp = c[i, i]
        fp = c[i, :].sum() - tp

        p[i] = tp/(tp + fp) if (tp + fp) > 0 else -1.0/c.shape[1]

    return p

def get_recall(c):

    r = np.zeros(c.shape[1])

    for i in range(c.shape[1]):

        tp = c[i, i]
        fn = c[:, i].sum() - tp

        r[i] = tp/(tp + fn)

    return r

# p: Prediction numpy array: one-hot encoded
# y: Groundtruth numpy array: one-hot encoded
# class_labels: Strings denoting the class label names
def plot_stats(p, y, class_labels = None, title = "stats", path = None):

    c, c_counts = confusion_matrix(p, y)

    pr = get_precission(c)
    rc = get_recall(c)

    f = plt.figure(figsize = (10, 5), dpi = 300)

    gs = f.add_gridspec(5, 2, width_ratios = (1, 1), height_ratios = (1, 5/4, 5/4, 5/4, 5/4), bottom = 0.1, top = 1.0, hspace = 0.3, wspace = 0.2)

    a_cnf = f.add_subplot(gs[1:, 0])
    a_rc  = f.add_subplot(gs[3:,  1])
    a_pr  = f.add_subplot(gs[1:3, 1], sharex = a_rc)
    a_hst = f.add_subplot(gs[0, 0], sharex = a_cnf)
    a_tot = f.add_subplot(gs[0, 1])

    if class_labels is None:

        a_cnf.set_xticks([])
        a_cnf.set_yticks([])

        a_rc.set_xticks([])
    else:

        a_cnf.set_xticks(np.arange(c.shape[1]), labels = class_labels, fontsize = 5, rotation = 45)
        a_cnf.set_yticks(np.arange(c.shape[0]), labels = class_labels, fontsize = 5)

        a_rc.set_xticks(np.arange(c.shape[1]), labels = class_labels, fontsize = 5, rotation = 45)

    a_rc.set_xlabel("Groundtruth class")
    a_cnf.set_xlabel("Groundtruth class")
    a_cnf.set_ylabel("Predicted class")
    a_hst.set_ylabel("Class sample\ncount")

    a_rc.bar(np.arange(c.shape[1]), rc, label = "Recall",     zorder = 2, color = "orange")
    a_rc.set_ylim(0.0, 1.0)
    a_pr.bar(np.arange(c.shape[1]), pr, label = "Precission", zorder = 2)
    a_pr.set_ylim(-1.0/c.shape[1], 1.0)

    a_rc.legend(fontsize = 5)
    a_rc.grid(color = "black", linewidth = 0.5, zorder = 1, alpha = 0.5)

    a_pr.legend(fontsize = 5)
    a_pr.grid(color = "black", linewidth = 0.5, zorder = 1, alpha = 0.5)

    a_cnf.imshow(c, aspect = "auto")
    a_cnf.hlines(np.arange(-0.5, c.shape[0]), -0.5, c.shape[0] - 0.5, color = "black", linewidth = 0.5)
    a_cnf.vlines(np.arange(-0.5, c.shape[0]), -0.5, c.shape[0] - 0.5, color = "black", linewidth = 0.5)
    a_cnf.plot(np.arange(c.shape[0]), np.arange(c.shape[1]), color = "red", linestyle = "dashed")

    a_tot.bar(1, rc.mean(), zorder = 2, color = "orange")
    a_tot.bar(3, pr[pr >= 0.0].mean(), zorder = 2)
    a_tot.grid(zorder = 1, alpha = 0.5, color = "black")
    a_tot.set_ylim(0.0, 1.0)

    a_tot.set_xticks([1, 3], labels = ["Macro recall: {:.2f}".format(rc.mean()), "Macro precission: {:.2f}".format(pr[pr >= 0.0].mean())], fontsize = 5)

    plt.setp(a_hst.get_xticklabels(), visible = False)
    plt.setp(a_pr.get_xticklabels(),  visible = False)

    a_hst.bar(np.arange(c.shape[1]), np.squeeze(c_counts))

    path = f"{title}.png" if path is None else join(path, f"{title}.png")

    f.savefig(path, bbox_inches = "tight")

    plt.close(f)

