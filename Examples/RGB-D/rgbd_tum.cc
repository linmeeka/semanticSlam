/**
* This file is a modified version of ORB-SLAM2.<https://github.com/raulmur/ORB_SLAM2>
*
* This file is part of DynaSLAM.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <unistd.h>
#include<opencv2/core/core.hpp>

#include "Geometry.h"
#include "MaskNet.h"
#include <System.h>


using namespace std;

std::vector<cv::Scalar> labelColor={
        /*0*/    cv::Scalar(0,0,0),cv::Scalar(139,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
                cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
        /*10*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
                cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
        /*20*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
                cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
        /*30*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
                cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
        /*40*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
                cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
        /*50*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
                cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
        /*60*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,128),cv::Scalar(0,0,0),
                cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,128,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
        /*70*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
                cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
        
        /*80*/    cv::Scalar(0,0,0)
            };

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

void getMask(const cv::Mat & maskRCNN, cv::Mat &mask, cv::Mat &maskColor);

void print(cv::Mat &mask)
{
     for(int i=0;i<mask.rows;i++)
    {
        for(int j=0;j<mask.cols;j++)
        {
            if(mask.channels()==1)
                cout<<mask.at<uchar>(i,j)<<" ";
            else
            {
                cout<<mask.at<cv::Vec3b>(i,j)[0]<<",";
                cout<<mask.at<cv::Vec3b>(i,j)[1]<<",";
                cout<<mask.at<cv::Vec3b>(i,j)[2]<<",";
                cout<<" ";
                
            }
        }
        cout<<endl;
    }
}

int main(int argc, char **argv)
{
    if(argc != 5 && argc != 6 && argc != 7)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association (path_to_masks) (path_to_output)" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    std::cout << "nImages: " << nImages << std::endl;

    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Initialize Mask R-CNN
    DynaSLAM::SegmentDynObject *MaskNet;
    if (argc==6 || argc==7)
    {
        cout << "Loading Mask R-CNN. This could take a while..." << endl;
        MaskNet = new DynaSLAM::SegmentDynObject();
        cout << "Mask R-CNN loaded!" << endl;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Dilation settings
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                           cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                           cv::Point( dilation_size, dilation_size ) );

    if (argc==7)
    {
        std::string dir = string(argv[6]);
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        dir = string(argv[6]) + "/rgb/";
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        dir = string(argv[6]) + "/depth/";
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        dir = string(argv[6]) + "/mask/";
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }

    // Main loop
        cv::Mat imRGB, imD;
        cv::Mat imRGBOut, imDOut,maskOut;

    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        //cout<<"channels: "<<imRGB.channels()<<endl;
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);
        //cout<<"channels: "<<imD.channels()<<endl;
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Segment out the images
        //cv::Mat mask = cv::Mat::ones(480,640,CV_8U);
        cv::Mat mask,maskColor;
        std::vector<cv::Rect> ROIRes;
        std::vector<int> ClassIdRes;
        if (argc == 6 || argc == 7)
        {
            cv::Mat maskRCNN;
            string segDir=vstrImageFilenamesRGB[ni].replace(0,4,"");
            segDir=segDir.replace(segDir.length()-4,4,"");
            MaskNet->GetSegmentation(imRGB,maskRCNN,ROIRes,ClassIdRes,string(argv[5]),segDir);
            //MaskNet->GetSegmentation(imRGB,maskRCNN,ROIRes,ClassIdRes,string(argv[5]),vstrImageFilenamesRGB[ni].replace(0,4,""));
            //maskRCNN = MaskNet->GetSegmentation(imRGB,string(argv[5]),vstrImageFilenamesRGB[ni].replace(0,4,""));
            // cv::Mat maskRCNNdil = maskRCNN.clone();
            // cv::dilate(maskRCNN,maskRCNNdil, kernel);
            // mask = mask - maskRCNNdil;
            //cv::Mat mask,maskColor;
            //mask = cv::Mat::ones(480,640,CV_8U);
            mask=maskRCNN.clone();
            maskColor=imRGB.clone();
            getMask(maskRCNN,mask,maskColor);
            //print(mask);
            //print(maskColor);
            //imwrite("mask.jpg",mask);
            //imwrite("maskcolor.jpg",maskColor);
            cv::Mat maskRCNNdil = mask.clone();
            //cv::dilate(mask,maskRCNNdil, kernel);
            cv::dilate(maskRCNN,maskRCNNdil, kernel);
            //imwrite("maskRCNNdil.jpg",maskRCNNdil);
            cv::Mat maskTmp = cv::Mat::ones(480,640,CV_8U);
            mask = maskTmp - maskRCNNdil;
           // imwrite("maskt.jpg",mask);
        }

        // Pass the image to the SLAM system
        //if (argc == 7){SLAM.TrackRGBD(imRGB,imD,mask,tframe,imRGBOut,imDOut,maskOut);}
        if (argc == 7){
           
            SLAM.TrackRGBD(imRGB,imD,mask,maskColor,ROIRes,ClassIdRes,tframe,imRGBOut,imDOut,maskOut);
            }
        else {SLAM.TrackRGBD(imRGB,imD,mask,tframe);}

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        if (argc == 7)
        {
            cv::imwrite(string(argv[6]) + "/rgb/" + vstrImageFilenamesRGB[ni],imRGBOut);
            vstrImageFilenamesD[ni].replace(0,6,"");
            cv::imwrite(string(argv[6]) + "/depth/" + vstrImageFilenamesD[ni],imDOut);
            cv::imwrite(string(argv[6]) + "/mask/" + vstrImageFilenamesRGB[ni],maskOut);
        }

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void getMask(const cv::Mat & maskRCNN, cv::Mat &mask, cv::Mat &maskColor)
{
    //cv::Scalar color(139,0,0);
    for(int i=0;i<maskRCNN.rows;i++)
    {
        for(int j=0;j<maskRCNN.cols;j++)
        {
            int val=maskRCNN.at<uchar>(i,j);
            if(val==0)
                continue;
            else
            {
                mask.at<uchar>(i,j)=1;
                cv::Scalar color=labelColor[val];
                maskColor.at<cv::Vec3b>(i,j)[0]=color[0];
                maskColor.at<cv::Vec3b>(i,j)[1]=color[1];
                maskColor.at<cv::Vec3b>(i,j)[2]=color[2];
                // if(maskRCNN.at<uchar>(i,j)==1)
                // {
                //     maskColor.at<cv::Vec3b>(i,j)[0]=color[0];
                //     maskColor.at<cv::Vec3b>(i,j)[1]=color[1];
                //     maskColor.at<cv::Vec3b>(i,j)[2]=color[2];
                // }
                // if(maskRCNN.at<uchar>(i,j)==2)
                // {
                //     maskColor.at<cv::Vec3b>(i,j)[0]=0;
                //     maskColor.at<cv::Vec3b>(i,j)[1]=128;
                //     maskColor.at<cv::Vec3b>(i,j)[2]=0;
                // }
                // if(maskRCNN.at<uchar>(i,j)==3)
                // {
                //     maskColor.at<cv::Vec3b>(i,j)[0]=0;
                //     maskColor.at<cv::Vec3b>(i,j)[1]=0;
                //     maskColor.at<cv::Vec3b>(i,j)[2]=128;
                // }
            }
        }
    }
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}
