#!/bin/bash
#Uncomment below line to run in benchmark mode
#python run.py -mode benchmarking -max_window_size 5 -dir "/Users/rmanchur/Documents/MAI_Thesis/data/" -fitsdir "/Users/rmanchur/Documents/MAI_Thesis/data/all_data/all_sets/" -targets "./data/points_to_check/target_list_small.txt" -vizualize_data True

#Uncomment below line to run in K-nn mode
#python run.py -mode knn -dtw_window_size 5 -nearest_neighbors 3  -dir "/Users/rmanchur/Documents/MAI_Thesis/data/" -fitsdir "/Users/rmanchur/Documents/MAI_Thesis/data/all_data/all_sets/" -targets "./data/points_to_check/target_list_small.txt" -out "nn_small_results.txt"

#Uncomment below line to run Hierarchical clustering with full processing on dataset
#python run.py -mode clustering -cluster_type hierarchical  -linkage_type complete -dir "/Users/rmanchur/Documents/MAI_Thesis/data/" -fitsdir "/Users/rmanchur/Documents/MAI_Thesis/data/all_data/renamed/" -targets "./data/points_to_check/targetlist.txt" -saved False -csv_file "full_coordinates_matrix.csv"

#Uncomment below line to run Hierarchical clustering with reading pre-processed coordinate matrix
#python run.py -mode clustering -cluster_type hierarchical  -linkage_type complete -dir "/Users/rmanchur/Documents/MAI_Thesis/data/" -saved True -csv_file "full_coordinates_matrix.csv"

#Uncomment below line to run k-means clustering with full processing on dataset
#python run.py -mode clustering -cluster_type k-means  -num_clusters 8 -dir "/Users/rmanchur/Documents/MAI_Thesis/data/" -fitsdir "/Users/rmanchur/Documents/MAI_Thesis/data/all_data/all_sets/" -targets "./data/points_to_check/target_list_small.txt"  -saved False -csv_file "coordinates_matrix.csv"
#python run.py -mode clustering -cluster_type k-means  -num_clusters 8 -dir "/Users/rmanchur/Documents/MAI_Thesis/data/" -fitsdir "/Users/rmanchur/Documents/MAI_Thesis/data/all_data/renamed/" -targets "./data/points_to_check/targetlist.txt"  -saved False -csv_file "full_coordinates_matrix_sqrt.csv"


#Uncomment below line to run k-means clustering with reading pre-processed coordinate matrix
#python run.py -mode clustering -cluster_type k-means  -num_clusters 8 -dir "/Users/rmanchur/Documents/MAI_Thesis/data/"  -saved True -csv_file "full_coordinates_matrix.csv"

#Uncomment below line to run DBA k-means clustering with full processing on dataset
#python run.py -mode clustering -cluster_type dtw  -num_clusters 9 -dir "/Users/rmanchur/Documents/MAI_Thesis/data/" -fitsdir "/Users/rmanchur/Documents/MAI_Thesis/data/all_data/renamed/" -targets "./data/points_to_check/targetlist.txt"  -saved False -csv_file "DBA_V2low.csv" -meassurement_type V2 -wavelength low -out "DBA_V2low"
#python run.py -mode clustering -cluster_type dtw  -num_clusters 9 -dir "/Users/rmanchur/Documents/MAI_Thesis/data/" -fitsdir "/Users/rmanchur/Documents/MAI_Thesis/data/all_data/renamed/" -targets "./data/points_to_check/targetlist.txt"  -saved False -csv_file "DBA_V2medium.csv" -meassurement_type V2 -wavelength medium -out "DBA_V2medium"
#python run.py -mode clustering -cluster_type dtw  -num_clusters 9 -dir "/Users/rmanchur/Documents/MAI_Thesis/data/" -fitsdir "/Users/rmanchur/Documents/MAI_Thesis/data/all_data/renamed/" -targets "./data/points_to_check/targetlist.txt"  -saved False -csv_file "DBA_V2high.csv" -meassurement_type V2 -wavelength high -out "DBA_V2high"
#python run.py -mode clustering -cluster_type dtw  -num_clusters 9 -dir "/Users/rmanchur/Documents/MAI_Thesis/data/" -fitsdir "/Users/rmanchur/Documents/MAI_Thesis/data/all_data/renamed/" -targets "./data/points_to_check/targetlist.txt"  -saved False -csv_file "DBA_CPlow.csv" -meassurement_type CP -wavelength low -out "DBA_CPlow"
#python run.py -mode clustering -cluster_type dtw  -num_clusters 9 -dir "/Users/rmanchur/Documents/MAI_Thesis/data/" -fitsdir "/Users/rmanchur/Documents/MAI_Thesis/data/all_data/renamed/" -targets "./data/points_to_check/targetlist.txt"  -saved False -csv_file "DBA_CPmedium.csv" -meassurement_type CP -wavelength medium -out "DBA_CPmedium"
#python run.py -mode clustering -cluster_type dtw  -num_clusters 9 -dir "/Users/rmanchur/Documents/MAI_Thesis/data/" -fitsdir "/Users/rmanchur/Documents/MAI_Thesis/data/all_data/renamed/" -targets "./data/points_to_check/targetlist.txt"  -saved False -csv_file "DBA_CPhigh.csv" -meassurement_type CP -wavelength high -out "DBA_CPhigh"

#Uncomment below line to run DBA k-means clustering with with reading pre-processed dataset
#python run.py -mode clustering -cluster_type dtw  -num_clusters 5 -dir "/Users/rmanchur/Documents/MAI_Thesis/data/"  -saved True -csv_file "DBA_CPhigh.csv" -out "DBA_CPhigh"




