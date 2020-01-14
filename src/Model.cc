#include "Model.h"

const int HEIGHT=480;
const int WIDTH=640;

Model::Model(const int index, const long unsigned int kf_index, const std::shared_ptr<SegData>& segData,const cv::Mat &Mask)
{
    mIndex = index;
    mvKeyframeIndexes.push_back(kf_index);
    mmKeyFrameObjectInfo.insert(std::make_pair(kf_index,segData));
    mClassId = segData->classId;
    isMoving=segData->weight>0?true:false;
    lastMask=Mask;
    matched=false;
    model = boost::make_shared<PointCloud>();
}

void Model::UpdateObjectInfo(const std::shared_ptr<SegData>& segData, const long unsigned int kf_index, const cv::Mat &Mask)
{
    mmKeyFrameObjectInfo.insert(std::make_pair(kf_index,segData));
    mvKeyframeIndexes.push_back(kf_index);
    lastMask=Mask;
    matched=true;
}

void Model::UpdatePointCloud(KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth, cv::Mat& mask)
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

pcl::PointCloud<PointT >::Ptr Model::GetIncrementModel(KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth, cv::Mat& mask)
{
    PointCloud::Ptr tmp( new PointCloud() );
    cv::Rect RoI=GetLastRect();
    // point cloud is null ptr
    // 3*3的像素区域取一个点
    int w=min(WIDTH-1,RoI.x+RoI.width);
    int h=min(HEIGHT-1,RoI.y+RoI.height);
    int x0=max(0,RoI.x);
    int y0=max(0,RoI.y);
    int step=2;
    //for ( int m=x0; m<w; m+=3 )
    for ( int n=RoI.x; n<RoI.x+RoI.width; n+=step )
    {
        //for ( int n=y0; n<h; n+=3 )
        for(int m=RoI.y;m<RoI.y+RoI.height;m+=step)
        {
            if(m>=HEIGHT || n>=WIDTH)   break;
            if(mask.ptr<uchar>(m)[n]!=mClassId)
                continue;
            mask.ptr<uchar>(m)[n]=255;
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
    cv::rectangle(mask, RoI, 255,1);
    
    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;

    //cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}