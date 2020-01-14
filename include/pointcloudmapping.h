/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"
#include <vector>
#include <queue>
#include "ModelManager.h"
#include <iostream>
#include <condition_variable>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

// for clustering
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <iostream>

typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointNCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;
typedef pcl::PointCloud<pcl::PointXYZL> pointcloudL;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

using namespace ORB_SLAM2;

class PointCloudMapping
{
public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    enum cloudType
    {RAW = 0, FILTERED = 1, REMOVAL = 2, CLUSTER = 3};
    
    PointCloudMapping(double resolution_);
    std::vector<cv::Scalar> colors;
    // 插入一个keyframe，会更新一次地图
    void insertKeyFrame( KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& mask, std::vector<std::shared_ptr<SegData>> &segDatas);
    void shutdown();
    void viewer();

    
protected:
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& mask);
    PointCloud::Ptr globalMap; // background
    PointCloud::Ptr background;
    shared_ptr<thread>  viewerThread;   
    
    bool    shutDownFlag    =false;
    mutex   shutDownMutex;  
    
    condition_variable  keyFrameUpdated;
    mutex               keyFrameUpdateMutex;
    
    // data to generate point clouds
    // vector<KeyFrame*>       keyframes;
    // vector<cv::Mat>         colorImgs;
    // vector<cv::Mat>         depthImgs;
    // vector<cv::Mat>         maskImgs;
    queue<KeyFrame*>       keyframes;
    queue<cv::Mat>         colorImgs;
    queue<cv::Mat>         depthImgs;
    queue<cv::Mat>         maskImgs;
    queue<std::vector<std::shared_ptr<SegData>>>  segDataQue;
    mutex                   keyframeMutex;
    uint16_t                lastKeyframeSize =0;

    double resolution = 0.01;
    pcl::VoxelGrid<PointT>  voxel;
    pcl::StatisticalOutlierRemoval<PointT> sor;// 创建滤波器对象

    ModelManager modelManager;

};

#endif // POINTCLOUDMAPPING_H
