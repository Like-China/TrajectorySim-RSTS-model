import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from trainer.models import EncoderDecoder
from loader.data_loader import DataLoader
import settings, time, os, shutil, logging, h5py
from tqdm import tqdm
from torch.backends import cudnn



def nll_criterion(vocab_size):
    """
    带权的NLLLoss损失函数， 将编码为0的填充位置的权重置为0
    :param vocab_size:
    :return: 带权损失函数
    """
    weight = torch.ones(vocab_size)
    weight[settings.PAD] = 0
    # sum-loss求和 null-loss取平均值 none显示全部loss
    criterion = nn.NLLLoss(weight, reduction='sum')
    return criterion


def kl_criterion(output, target, criterion, V, D):
    """
    output (batch, vocab_size)  128*18866
    target (batch,)  128*1
    criterion (nn.KLDIVLoss)
    V (vocab_size, k) 18866*10
    D (vocab_size, k) 18866*10

    该评价模型评价每一批128个目标cell的10个邻居对应的输出权重与真实权重的距离 128*10

    只考虑每个点的10个邻居
    """
    # 获取128个目标cell的10个邻居
    # 第一个参数是索引的对象，第二个参数0表示按行索引，1表示按列进行索引，第三个参数是一个tensor，就是索引的序号
    indices = torch.index_select(V, 0, target)
    # 收集输出的128个目标对应的10个邻居的权重，是模型预测出来的权重
    outputk = torch.gather(output, 1, indices)
    # 获取128个目标cell的10个邻居对应的权重，从D中获取，是真实权重
    targetk = torch.index_select(D, 0, target)
    return criterion(outputk, targetk)


def dist2weight(D, dist_decay_speed=0.8):
    """
    对于K个邻居，按照距离大小给出权重，公式5中的W

    :param D: 和topK邻居的距离矩阵
    :param dist_decay_speed: 衰减指数
    :return:
    """
    # D = D.div(100)
    D = torch.exp(-D * dist_decay_speed)
    s = D.sum(dim=1, keepdim=True)
    D = D / s
    # The PAD should not contribute to the decoding loss
    D[settings.PAD, :] = 0.0
    return D


def genLoss(gen_data, m0, m1, loss_function, args):
    """
    计算一批训练数据的损失

    :param gen_data:
    :param m0: encoder-decoder
    :param m1: 获取中间层的表征向量
    :param loss_function: 损失函数
    :param args:
    :return:
    """
    # src (seq_len1, batch), lengths (1, batch), trg (seq_len2, batch)
    input, lengths, target = gen_data.src, gen_data.lengths, gen_data.trg
    if args.cuda and torch.cuda.is_available():
        input, lengths, target = input.cuda(), lengths.cuda(), target.cuda()
    #  (seq_len2, batch, hidden_size)
    output = m0(input, lengths, target)
    
    batch = output.size(1)
    loss = 0
    #  we want to decode target in range [BOS+1:EOS]
    target = target[1:]
    # args.generator_batch 32每一次生成的words数目，要求内存
    # output [max_target_szie, 128, 256]
    for o, t in zip(output.split(args.generator_batch),
                    target.split(args.generator_batch)):
        # (generator_batch, batch, hidden_size) => (batch*generator_batch, hidden_size)
        o = o.view(-1, o.size(2))
        o = m1(o)
        #  (batch*generator_batch,)
        t = t.view(-1)
        loss += loss_function(o, t)
    return loss.div(batch)


def disLoss(a, p, n, m0, triplet_loss, args):
    """
    计算相似性损失，即三角损失

    通过a,p,n三组轨迹，经过前向encoder,接着通过encoder_hn2decoder_h0，取最后一层向量作为每组每个轨迹的代表

    a (named tuple): anchor data
    p (named tuple): positive data
    n (named tuple): negative data
    """
    # a_src (seq_len, 128)
    a_src, a_lengths, a_invp = a.src, a.lengths, a.invp
    p_src, p_lengths, p_invp = p.src, p.lengths, p.invp
    n_src, n_lengths, n_invp = n.src, n.lengths, n.invp
    if args.cuda and torch.cuda.is_available():
        a_src, a_lengths, a_invp = a_src.cuda(), a_lengths.cuda(), a_invp.cuda()
        p_src, p_lengths, p_invp = p_src.cuda(), p_lengths.cuda(), p_invp.cuda()
        n_src, n_lengths, n_invp = n_src.cuda(), n_lengths.cuda(), n_invp.cuda()
    # (num_layers * num_directions, batch, hidden_size)  (2*3, 128, 256/2)
    a_h, _ = m0.encoder(a_src, a_lengths)
    p_h, _ = m0.encoder(p_src, p_lengths)
    n_h, _ = m0.encoder(n_src, n_lengths)
    # (num_layers, batch, hidden_size * num_directions) (3,128,256)
    a_h = m0.encoder_hn2decoder_h0(a_h)
    p_h = m0.encoder_hn2decoder_h0(p_h)
    n_h = m0.encoder_hn2decoder_h0(n_h)
    # take the last layer as representations (batch, hidden_size * num_directions) (128,256)
    a_h, p_h, n_h = a_h[-1], p_h[-1], n_h[-1]
    return triplet_loss(a_h[a_invp], p_h[p_invp], n_h[n_invp])  # (128,256)


def init_parameters(model):
    # 模型参数初始化
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint)
    if is_best:
        shutil.copyfile(args.checkpoint, os.path.join(args.data, 'best_model.pt'))


def validate(val_data, model, loss_function, args):
    """
    验证获取genLoss, 得到验证集的损失

    val_data (DataLoader)
    """
    m0, m1 = model
    # switch to evaluation mode
    m0.eval()
    m1.eval()

    # 获取验证集迭代的批次数目
    num_iteration = val_data.size // args.batch
    if val_data.size % args.batch > 0:
        num_iteration += 1

    total_gen_loss = 0
    for iteration in tqdm(range(num_iteration), desc="验证检验"):
        # 获取一批验证集数据
        gen_data = val_data.get_batch_generative()
        with torch.no_grad():
            gen_loss = genLoss(gen_data, m0, m1, loss_function, args)
            total_gen_loss += gen_loss.item() * gen_data.trg.size(1)
    # switch back to training mode
    m0.train()
    m1.train()
    return total_gen_loss / val_data.size


def load_data(args):
    # 加载训练集，包装到DataLoader中
    train_src = os.path.join(args.data, "train.src")
    train_trg = os.path.join(args.data, "train.trg")
    train_mta = os.path.join(args.data, "train.mta")
    train_data = DataLoader(train_src, train_trg, train_mta, args.batch, args.bucket_size)
    print("Reading training data...")
    train_data.load(args.read_train_num)
    print("trajectory num: %d maxId: %d minId: %d" % (train_data.size, train_data.maxID, train_data.minID))
    print("Allocation: {}".format(train_data.allocation))
    print("Percent: {}".format(train_data.p))

    # load validate data if exists
    val_src = os.path.join(args.data, "val.src")
    val_trg = os.path.join(args.data, "val.trg")
    val_mta = os.path.join(args.data, "val.mta")
    val_data = None
    if os.path.isfile(val_src) and os.path.isfile(val_trg):
        val_data = DataLoader(val_src, val_trg, val_mta, args.batch, args.bucket_size, True)
        print("Reading validation data...")
        val_data.load(args.read_val_num)
        print("trajectory num: %d maxId: %d minId: %d"% (val_data.size, val_data.maxID, val_data.minID))
        assert val_data.size > 0, "Validation data size must be greater than 0"
        print("Loaded validation data size {}".format(val_data.size))
    else:
        print("No validation data found, training without validating...")
    return train_data, val_data


def set_loss(args):
    """
    设置KL散度损失函数
    
    :param args:  参数设定
    :return: 设置的损失函数
    """
    if args.criterion_name == "NLL":
        criterion = nll_criterion(args.vocab_size)
        if args.cuda and torch.cuda.is_available():
            criterion.cuda()
        return lambda o, t: criterion(o, t)
    else:
        assert os.path.isfile(args.knn_vocabs), "{} does not exist".format(args.knn_vocabs)
        print("Loading vocab distance file {}...".format(args.knn_vocabs))
        with h5py.File(args.knn_vocabs, 'r') as f:
            # VD size = (vocal_size, 10) 第i行为第i个轨迹与其10个邻居
            V, Ds, Dt = f["V"][...], f["Ds"][...], f["Dt"][...]
            V, Ds, Dt = torch.LongTensor(V), torch.FloatTensor(Ds), torch.FloatTensor(Dt)
            Ds, Dt = dist2weight(Ds, args.dist_decay_speed),dist2weight(Dt, args.dist_decay_speed)
            D = (1-args.timeWeight)*Ds + args.timeWeight*Dt
        if args.cuda and torch.cuda.is_available():
            V, D = V.cuda(), D.cuda()
        criterion = nn.KLDivLoss(reduction='sum')
        if args.cuda and torch.cuda.is_available():
            criterion.cuda()
        return lambda o, t: kl_criterion(o, t, criterion, V, D)


def train(args):
    """
    正式的训练过程

    :param args: 参数设定
    :return: None
    """
    logging.basicConfig(filename=os.path.join(args.data, "training.log"), level=logging.INFO)
    train_data, val_data = load_data(args)
    # 创建损失函数，模型以及最优化训练
    loss_function = set_loss(args)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # 输入到输出整个encoder-decoder的map
    m0 = EncoderDecoder(args)
    #  Encoder到Decoder 的中间输出输出到词汇表向量的映射，并进行了log操作
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size), nn.LogSoftmax(dim=1))
    if args.cuda and torch.cuda.is_available():
        print("=> training with GPU")
        # m0 = torch.nn.DataParallel(m0, device_ids=settings.ids).cuda()
        # m1 = torch.nn.DataParallel(m1, device_ids=settings.ids).cuda()
        # m0 = nn.DataParallel(m0, dim=1)
        # m1 = nn.DataParallel(m1)
        m0.cuda()
        m1.cuda()
    else:
        print("=> training with CPU")
    m0_optimizer = torch.optim.Adam(m0.parameters(), lr=args.learning_rate)
    m1_optimizer = torch.optim.Adam(m1.parameters(), lr=args.learning_rate)

    # 加载模型状态和优化器状态，如果存在已经保存的训练状态，如果不存在则重新开始生成
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        logging.info("Restore training @ {}".format(time.ctime()))
        checkpoint = torch.load(args.checkpoint)
        args.start_iteration = checkpoint["iteration"]
        best_train_gen_loss = checkpoint["best_train_gen_loss"]
        best_train_dis_loss = checkpoint["best_train_dis_loss"]
        best_train_loss = checkpoint["best_train_loss"]
        best_val_loss = checkpoint["best_val_loss"]

        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        m0_optimizer.load_state_dict(checkpoint["m0_optimizer"])
        m1_optimizer.load_state_dict(checkpoint["m1_optimizer"])
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        logging.info("Start training @ {}".format(time.ctime()))
        best_train_gen_loss = float('inf')
        best_train_dis_loss = float('inf')
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        print("=> initializing the parameters...")
        init_parameters(m0)
        init_parameters(m1)

    num_iteration = args.iter_num*args.epochs
    print("开始训练："+str(time.ctime()))
    print("Iteration starts at {} and will end at {} \n".format(args.start_iteration, num_iteration-1))
    invalid_count = 0  # 用一个计数器计数测试集损失未下降的次数，若超过一定次数，则直接停止训练
    for iteration in range(args.start_iteration+1, num_iteration):
        try:
            # 梯度初始化为0
            m0_optimizer.zero_grad()
            m1_optimizer.zero_grad()
            # 获取一批补位+转置后的数据对象 TF=['src', 'lengths', 'trg', 'invp']
            # src (seq_len1, batch), lengths (1, batch), trg (seq_len2, batch)
            gen_data = train_data.get_batch_generative()
            # 计算生成损失+三元判别损失
            train_gen_loss = genLoss(gen_data, m0, m1, loss_function, args)
            train_dis_cross, train_dis_inner = torch.tensor(0), torch.tensor(0)
            if args.use_discriminative and iteration % args.dis_freq == 0:
                # a和p的轨迹更接近 a.src.size = [max_length,128]
                a, p, n = train_data.get_apn_cross()
                train_dis_cross = disLoss(a, p, n, m0, triplet_loss, args)
                # a,p,n是由同一组128个轨迹采样得到的新的128个下采样轨迹集合
                a, p, n = train_data.get_apn_inner()
                train_dis_inner = disLoss(a, p, n, m0, triplet_loss, args)
            # 损失按一定权重相加 train_gen_loss： 使损失尽可能小 discriminative——loss: 使序列尽可能相似
            # 计算词的平均损失
            train_gen_loss = train_gen_loss / gen_data.trg.size(0)
            train_dis_loss = train_dis_cross + train_dis_inner
            train_loss = (1-args.discriminative_w)*train_gen_loss + args.discriminative_w * train_dis_loss
            train_loss.backward()
            # limit the clip of each grad
            clip_grad_norm_(m0.parameters(), args.max_grad_norm)
            clip_grad_norm_(m1.parameters(), args.max_grad_norm)
            m0_optimizer.step()
            m1_optimizer.step()
            if iteration % args.print_freq == 0:
                print("\n\ncurrent time:"+str(time.ctime()))
                print("Iteration: {0:}\nTrain Generative Loss: {1:.3f}\nTrain Discriminative Cross Loss: {2:.3f}"
                      "\nTrain Discriminative Inner Loss: {3:.3f}\nTrain Loss: {4:.3f}"\
                      .format(iteration, train_gen_loss, train_dis_cross, train_dis_inner, train_loss))
                print("best_train_loss: %.3f" % best_train_loss)
                print("best_train_dis_loss: %.3f" % best_train_dis_loss)
                print("best_val_loss: %.3f" % best_val_loss)
                
            # 定期存储训练状态，通过验证集前向计算当前模型损失，若能获得更小损失，则保存最新的模型参数
            # 只有训练集模型变好，才进行测试机检验和存储
            if iteration % args.save_freq == 0 and iteration > 0:
                # 如果训练集能够取得更好的模型，再进一步进行验证集验证
                if train_loss > best_train_loss:
                    invalid_count += 1
                    continue
                val_loss = validate(val_data, (m0, m1), loss_function, args)
                print("current validate loss: ", val_loss)
                # 如果验证集也能取得更小的
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    invalid_count = 0
                    logging.info("Best model with loss {} at {}".format(best_val_loss, time.ctime()))
                    is_best = True
                else:
                    is_best = False
                    invalid_count += 1
                    if invalid_count > 10:
                        print("多次训练不能再减少损失，停止训练")
                        break
                if is_best:
                    # 更新训练集的损失
                    best_train_loss = train_loss.item()
                    if train_gen_loss < best_train_gen_loss:
                        best_train_gen_loss = best_train_loss
                    train_dis_loss = train_dis_loss.item()
                    if train_dis_loss < best_train_dis_loss:
                        best_train_dis_loss = train_dis_loss
                    print("Saving the model at iteration {} validation loss {}".format(iteration, val_loss))
                    save_checkpoint({
                        "iteration": iteration,
                        "best_train_gen_loss": best_train_gen_loss,
                        "best_train_dis_loss": best_train_dis_loss,
                        "best_train_loss": best_train_loss,
                        "best_val_loss": best_val_loss,
                        "m0": m0.state_dict(),
                        "m1": m1.state_dict(),
                        "m0_optimizer": m0_optimizer.state_dict(),
                        "m1_optimizer": m1_optimizer.state_dict(),
                    }, is_best, args)
        except KeyboardInterrupt:
            break
