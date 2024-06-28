"""Utility script to enable the use of the Cas13a predictive models from the ADAPT publication (https://www.nature.com/articles/s41587-022-01213-5)"""
import os
import adapt
from adapt.utils import predict_activity
import tensorflow as tf

dir_path = adapt.utils.version.get_project_path()
cla_path_all = os.path.join(dir_path, 'models', 'classify', 
                                        'cas13a')
reg_path_all = os.path.join(dir_path, 'models', 'regress',
                                        'cas13a')
cla_version = adapt.utils.version.get_latest_model_version(cla_path_all)
reg_version = adapt.utils.version.get_latest_model_version(reg_path_all)
cla_path = os.path.join(cla_path_all, cla_version)
reg_path = os.path.join(reg_path_all, reg_version)
pred = predict_activity.Predictor(cla_path, reg_path)

target_len = 48
guide_len = 28
pad_len = tf.constant((target_len - guide_len) / 2)

def run_full_model(gen_guide, target_set, model_type = 'both'):
    """
    Function to run the both the predictive and regression model from the ADAPT publication (https://www.nature.com/articles/s41587-022-01213-5)
    Args:
        gen_guide: List of guide sequences to be evaluated
        target_set: List of target sequences to be evaluate each guide sequence against
        model_type: Type of model to run. Run the regression model, classification model or both.
    Returns:
        pred_perf_list: List of predicted performance values for each guide-target pair
        classify_perf_list: List of predicted classification values for each guide-target pair
    """

    assert model_type in ['regress', 'both', 'classify']
    
    # Prepare the guide-target arrays for the pred model 
    pred_input_list  = [] 
 
    for guide in gen_guide:
        #print(f"Guide Length: {len(guide)}")
        for target in target_set:
            #print(f"Target Length: {len(target)}") 
            gen_guide_padded = tf.pad(guide, [[pad_len, pad_len], [0, 0]])
            pred_input_list.append(tf.concat([target, gen_guide_padded], axis=1))

    regression_output = pred.regression_model.call(pred_input_list, training = False)
    pred_perf_list = tf.split(tf.reshape(regression_output, [len(target_set) * len(gen_guide)]), len(gen_guide))

    classifier_output = pred.classification_model.call(pred_input_list, training = False)
    classify_perf_list = tf.split(tf.reshape(classifier_output, [len(target_set) * len(gen_guide)]), len(gen_guide))

    if(model_type == 'both'):
        return pred_perf_list, classify_perf_list
    elif(model_type == 'regress'):
        return pred_perf_list
    elif(model_type == 'classify'):
        return classify_perf_list