#include "Model.h"

Model::Model(const int index, const long unsigned int kf_index, const std::shared_ptr<SegData>& segData)
{
    mIndex = index;
    mvKeyframeIndexes.push_back(kf_index);
    mmKeyFrameObjectInfo.insert(std::make_pair(kf_index,segData));
    mClassId = segData->classId;
    isMoving=segData->IsMove;
}

void Model::UpdateObjectInfo(const std::shared_ptr<SegData>& segData, const long unsigned int kf_index)
{
    mmKeyFrameObjectInfo.insert(std::make_pair(kf_index,segData));
    mvKeyframeIndexes.push_back(kf_index);
}

void Model::UpdatePointCloud(KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth, const cv::Mat& mask)
{
    PointCloud::Ptr inc;
    inc = GetIncrementModel(kf, color, depth, mask);
    if(isMoving)
    {
        *(this->model) = *inc;
    }
    else
    {
        *(this->model) += *inc;
    }
}

pcl::PointCloud<PointT >::Ptr Model::GetIncrementModel(KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth, const cv::Mat& mask)
{
    PointCloud::Ptr tmp( new PointCloud() );
    cv::Rect RoI=GetLastRect();
    // point cloud is null ptr
    // 3*3的像素区域取一个点
    for ( int m=RoI.x; m<RoI.width; m+=3 )
    {
        for ( int n=RoI.y; n<RoI.height; n+=3 )
        {
            if(mask.ptr<uchar>(m)[n]!=mClassId)
                continue;
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>10)
                continue;
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
            tmp->points.push_back(p);
        }
    }

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;

    //cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}