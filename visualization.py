import os, sys
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

def affichage(forged_path, candidate_path, mask_path, output_dir="./output", frames_dir="./frames"):
    forgedName = os.path.basename(forged_path)

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    rows = 6
    columns = 6
    plt.clf()

    try:
        plt.subplot(rows/2, columns/2, 1), plt.imshow(mpimg.imread(forged_path))
        plt.title('Query'), plt.xticks([]), plt.yticks([])
    except Exception as e:
        print(e)
    try:
        plt.subplot(rows/2, columns/2, 2), plt.imshow(mpimg.imread(candidate_path))
        plt.title("Candidate"), plt.xticks([]), plt.yticks([])
    except Exception as e:
        print(e)
    try:
        plt.subplot(rows/2, columns/2, 3)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "forged", forgedName)))
        plt.title("Crop"), plt.xticks([]), plt.yticks([])
    except Exception as e:
        print(e)
    try:
        plt.subplot(rows/2, columns/2, 4)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "homography", forgedName)))
        plt.title('Homography (inliers)'), plt.xticks([]), plt.yticks([])
    except Exception as e:
        print(e)
    try:
        plt.subplot(rows, columns, 2*columns + 3)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "outliers", forgedName)))
        plt.title('Outliers'), plt.xticks([]), plt.yticks([])
    except Exception as e:
        print(e)
    try:
        plt.subplot(rows, columns, 2*columns + 4)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "outliers_cleared", forgedName)))
        plt.title('Outliers clear'), plt.xticks([]), plt.yticks([])
    except Exception as e:
        print(e)
    try:
        plt.subplot(rows, columns, 3*columns + 3)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "density", forgedName)))
        plt.title('Density map'), plt.xticks([]), plt.yticks([])
    except Exception as e:
        print(e)
    try:
        plt.subplot(rows/2, columns/2, 6), plt.imshow(mpimg.imread(mask_path))
        plt.title("GT"), plt.xticks([]), plt.yticks([])
    except Exception as e:
        print(e)
    try:
        plt.subplot(rows, columns, (4 * columns) + 1)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "morpho", forgedName)))
        plt.title('Morpho'), plt.xticks([]), plt.yticks([])
    except Exception as e:
        print(e)
    try:
        plt.subplot(rows, columns, (4 * columns) + 2)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "10", forgedName)))
        plt.title('0.1'), plt.xticks([]), plt.yticks([])
        plt.subplot(rows, columns, (4 * columns) + 3)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "30", forgedName)))
        plt.title('0.3'), plt.xticks([]), plt.yticks([])
        plt.subplot(rows, columns, (4 * columns) + 4)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "50", forgedName)))
        plt.title('0.5'), plt.xticks([]), plt.yticks([])
        plt.subplot(rows, columns, (4 * columns) + 5)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "70", forgedName)))
        plt.title('0.7'), plt.xticks([]), plt.yticks([])
        plt.subplot(rows, columns, (4 * columns) + 6)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "90", forgedName)))
        plt.title('0.9'), plt.xticks([]), plt.yticks([])
    except Exception as e:
        print(e)
    try:
        plt.subplot(rows, columns, (5 * columns) + 1)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "visualization", "redAndBlue", "morpho", forgedName)))
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.subplot(rows, columns, (5 * columns) + 2)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "visualization", "redAndBlue", "10", forgedName)))
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.subplot(rows, columns, (5 * columns) + 3)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "visualization", "redAndBlue", "30", forgedName)))
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.subplot(rows, columns, (5 * columns) + 4)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "visualization", "redAndBlue", "50", forgedName)))
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.subplot(rows, columns, (5 * columns) + 5)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "visualization", "redAndBlue", "70", forgedName)))
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.subplot(rows, columns, (5 * columns) + 6)
        plt.imshow(mpimg.imread(os.path.join(output_dir, "visualization", "redAndBlue", "90", forgedName)))
        plt.title(''), plt.xticks([]), plt.yticks([])
    except Exception as e:
        print(e)

    # plt.show()
    filename = 'frames/' + forgedName + '.png'
    print "savefig : ", filename
    plt.savefig(filename, format='png', dpi=750)
    plt.clf()
    print "END"
