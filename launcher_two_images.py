#!/usr/bin/python
import argparse
import functions
import visualization

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-f', help='path to the forged image')
    parser.add_argument('-c', help='path to the candidate image')
    parser.add_argument('-m', help='path to the mask image')
    args = parser.parse_args()

    print "Parameters : "
    print "Forged : ", args.f
    print "Candidate : ", args.c
    print "Mask : ", args.m
    print

    # Three parameters :
    # 1/ path to the forged image
    # 2/ path to the candidate image
    # 3/ [optional, default = './output'] path to the directory where the results will be saved
    functions.compare_images(args.f, args.c, output_directory="./output")

    # Five parameters :
    # 1/ path to the forged image
    # 2/ path to the candidate image
    # 3/ path to the candidate image
    # 4/ [optional, default = './output'] path to the directory where the results are saved
    # 5/ [optional, default = './frames'] path to the directory where the visualization will be saved
    visualization.affichage(args.f, args.c, args.m, output_dir="./output", frames_dir="./frames")