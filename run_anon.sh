#!/bin/bash
apptainer exec --bind $PWD,$PWD/bin/dicom_video_anonymization.sh:/opt/app/dicom_video_anonymization,/ceph/chpc/rcif_datasets/clinical_datasets/us dcm-video-tools.sif dicom_video_anonymization $@
