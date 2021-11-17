from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.models import Model

from nncompress import backend as M

def make_teacher_output(name, t_idx=0, flow_idx=0, tensor_idx=0):
    return (t_idx, name, flow_idx, tensor_idx)

def make_student_output(name, flow_idx=0, tensor_idx=0):
    return (name, flow_idx, tensor_idx)

class Distillery(object):

    def __init__(self, teachers, student):
        self.teachers = teachers
        self.student = student

    def prep(self, recipe):
        self._recipe = recipe # backup
        student = tf.keras.models.clone_model(self.student, input_tensors=self.student.input)

        # Extract intermediate features to be used in computing distillation losses.
        t_outputs = [{} for _ in range(len(self.teachers))]
        for idx, (teachers_outputs, student_outputs, weight, func) in enumerate(self._recipe):
            if type(teachers_outputs) == tuple:
                teachers_outputs = [teachers_outputs]
            for t_item in teachers_outputs:
                t_idx, name, flow_idx, tensor_idx = t_item
                tensors = self.teachers[t_idx].get_layer(name).outbound_nodes[flow_idx].output_tensors
                if type(tensors) != tf.Tensor:
                    tensors = tensors[tensor_idx]
                if (name, flow_idx, tensor_idx) not in t_outputs[t_idx]:
                    t_outputs[t_idx][(name, flow_idx, tensor_idx)] = tensors

        # Define teachers with intermediate features 
        teachers = []
        t_output_idx = [{} for _ in range(len(self.teachers))]
        for idx, teacher in enumerate(self.teachers):
            output = []
            for item, val in t_outputs[idx].items():
                output.append(val)
                t_output_idx[idx][item] = len(output)-1
            teachers.append(Model(inputs=teacher.input, outputs=output))
  
        # Compute loss
        for idx, (teachers_outputs, student_outputs, weight, func) in enumerate(self._recipe):
            if type(teachers_outputs) == tuple:
                teachers_outputs = [teachers_outputs]
            if type(student_outputs) == tuple:
                student_outputs = [student_outputs]

            t_tensors = []
            for t_item in teachers_outputs:
                t_idx, name, flow_idx, tensor_idx = t_item
                outputs = teachers[t_idx](student.input)
                t_tensors.append(outputs[t_output_idx[t_idx][(name, flow_idx, tensor_idx)]])
            s_tensors = []
            for s_item in student_outputs:
                name, flow_idx, tensor_idx = s_item 
                tensors = student.get_layer(name).outbound_nodes[flow_idx].output_tensors
                if type(tensors) != tf.Tensor:
                    tensors = tensors[tensor_idx]
                s_tensors.append(tensors)

            if len(t_tensors) == 1:
                t_tensors = t_tensors[0]
            if len(s_tensors) == 1:
                s_tensors = s_tensors[0]
            if type(func) == str:
                distill_loss = getattr(tf.keras.losses, func)(t_tensors, s_tensors)
            else:
                distill_loss = func(t_tensors, s_tensors)
            student.add_loss(weight * distill_loss)

        return student

if __name__ == "__main__":

    import tensorflow as tf
    from tensorflow import keras

    s_model = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling=None, classes=10)
    t_model = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling=None, classes=10)
    t2_model = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling=None, classes=10)

    t_model = M.add_prefix(t_model, "t_", (1, 128, 128, 3))
    t2_model = M.add_prefix(t2_model, "t2_", (1, 128, 128, 3))

    distiller = Distillery([t_model, t2_model], s_model)
    name = "conv2_block1_2_conv"

    student = distiller.prep([
        (
            make_teacher_output("t_"+name),
            make_student_output(name),
            1.0,
            "categorical_crossentropy"
        ),
        ( 
            make_teacher_output("t2_"+name, t_idx=1),
            make_student_output(name),
            1.0,
            "categorical_crossentropy"
        )
    ])

    tf.keras.utils.plot_model(student, "xx.png")
