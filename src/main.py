from argparse import ArgumentParser
import lab1_impl
import lab1_keras


parser = ArgumentParser()
parser.add_argument('--type', type=str, choices=['lab', 'keras'], default='lab',
                    help='Choosing network for run')
parser.add_argument('--hidden_num', type=int,  default=240,
                    help='Hidden layer nodes num')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Sample batch size')
parser.add_argument('--rate', type=float, default=0.3,
                    help='Learning rate')
parser.add_argument('--epochs', type=int, default=5,
                    help='Epochs count')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.type == 'lab':
        train_result, test_result, time = lab1_impl.run(args.hidden_num, args.batch_size, args.rate, args.epochs)
        print('Train loss: ', train_result[0])
        print('Train accuracy: ', train_result[1])
        print('Test loss: ', test_result[0])
        print('Test accuracy: ', test_result[1])
        print('Time: ', time)
    else:
        train_result, test_result, time = lab1_keras.run(args.hidden_num, args.batch_size, args.rate, args.epochs)
        print('Train loss: ', train_result[0])
        print('Train accuracy: ', train_result[1])
        print('Test loss: ', test_result[0])
        print('Test accuracy: ', test_result[1])
        print('Time: ', time)



