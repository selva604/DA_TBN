import torch
from torch import nn
from torch.nn.init import normal_, constant_
from context_gating import Context_Gating
from multimodal_gating import Multimodal_Gated_Unit
from ops.basic_ops import ConsensusModule
from torch.autograd import Function
import torch.nn.functional as F

class GradReverse(Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)
    @staticmethod
    def backward(self, grad_output):
        #pdb.set_trace()
        return (grad_output * -1.0)

def grad_reverse(x):
    return GradReverse.apply(x)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

class netD_dc(nn.Module):
    def __init__(self):
        super(netD_dc, self).__init__()
        self.fc1 = nn.Linear(2048,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        x = self.fc3(x)
        return x

class netD(nn.Module):
    def __init__(self,context=False):
        super(netD, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat#torch.cat((feat1,feat2),1)#F
        else:
          return x

class Fusion_Classification_Network(nn.Module):

    def __init__(self, feature_dim, modality, midfusion, num_class,
                 consensus_type, before_softmax, dropout, num_segments):
        super().__init__()
        self.num_class = num_class
        self.modality = modality
        self.midfusion = midfusion
        self.reshape = True
        self.consensus = ConsensusModule(consensus_type)
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.num_segments = num_segments
        #global
        #self.netD = netD()
        self.netD_dc = netD_dc()
        
        if not self.before_softmax:
            self.softmax = nn.Softmax()

        if len(self.modality) > 1:  # Fusion

            if self.midfusion == 'concat':
                self._add_audiovisual_fc_layer(len(self.modality) * feature_dim, 512)
                self._add_classification_layer(512)

            elif self.midfusion == 'context_gating':
                self._add_audiovisual_fc_layer(len(self.modality) * feature_dim, 512)
                self.context_gating = Context_Gating(512)
                self._add_classification_layer(512)

            elif self.midfusion == 'multimodal_gating':
                self.multimodal_gated_unit = Multimodal_Gated_Unit(feature_dim, 512)
                if self.dropout > 0:
                    self.dropout_layer = nn.Dropout(p=self.dropout)
                self._add_classification_layer(512)

        else:  # Single modality
            if self.dropout > 0:
                self.dropout_layer = nn.Dropout(p=self.dropout)

            self._add_classification_layer(feature_dim)

    def _add_classification_layer(self, input_dim):

        std = 0.001
        if isinstance(self.num_class, (list, tuple)):  # Multi-task

            self.fc_verb = nn.Linear(input_dim, self.num_class[0])
            self.fc_noun = nn.Linear(input_dim, self.num_class[1])
            normal_(self.fc_verb.weight, 0, std)
            constant_(self.fc_verb.bias, 0)
            normal_(self.fc_noun.weight, 0, std)
            constant_(self.fc_noun.bias, 0)
        else:
            self.fc_action = nn.Linear(input_dim, self.num_class)
            normal_(self.fc_action.weight, 0, std)
            constant_(self.fc_action.bias, 0)

    def _add_audiovisual_fc_layer(self, input_dim, output_dim):

        self.fc1 = nn.Linear(input_dim, output_dim)
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        std = 0.001
        normal_(self.fc1.weight, 0, std)
        constant_(self.fc1.bias, 0)
        self.relu = nn.ReLU()

    def forward(self, inputs):

        if len(self.modality) > 1:  # Fusion
            if self.midfusion == 'concat':
                base_out = torch.cat(inputs, dim=1)
                domain_p = self.netD_dc(grad_reverse(base_out))

                base_out = self.fc1(base_out)
                base_out = self.relu(base_out)
            elif self.midfusion == 'context_gating':
                base_out = torch.cat(inputs, dim=1)
                base_out = self.fc1(base_out)
                base_out = self.relu(base_out)
                base_out = self.context_gating(base_out)
            elif self.midfusion == 'multimodal_gating':
                base_out = self.multimodal_gated_unit(inputs)
        else:  # Single modality
            base_out = inputs[0]

        #Domainの予測
        #domain_p = self.netD_dc(grad_reverse(base_out))

        if self.dropout > 0:
            base_out = self.dropout_layer(base_out)

        # Snippet-level predictions and temporal aggregation with consensus
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            # Verb
            base_out_verb = self.fc_verb(base_out)
            if not self.before_softmax:
                base_out_verb = self.softmax(base_out_verb)
            if self.reshape:
                base_out_verb = base_out_verb.view((-1, self.num_segments) + base_out_verb.size()[1:])
            output_verb = self.consensus(base_out_verb)

            # Noun
            base_out_noun = self.fc_noun(base_out)
            if not self.before_softmax:
                base_out_noun = self.softmax(base_out_noun)
            if self.reshape:
                base_out_noun = base_out_noun.view((-1, self.num_segments) + base_out_noun.size()[1:])
            output_noun = self.consensus(base_out_noun)

            output = (output_verb.squeeze(1), output_noun.squeeze(1))

        else:
            base_out = self.fc_action(base_out)
            if not self.before_softmax:
                base_out = self.softmax(base_out)
            if self.reshape:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

            output = self.consensus(base_out)
            output = output.squeeze(1)
        #domain_p = self.netD_dc(grad_reverse(output))
        return output,domain_p
