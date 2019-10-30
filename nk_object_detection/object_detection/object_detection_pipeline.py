from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
import sys

# Create pipeline
pipeline_description = Pipeline()
pipeline_descripion.add_input(name = 'inputs')

# Step 0: Denormalize primitive
step_0 = PrimitiveStep(primitive = index.get_primitive('d3m.primitives.data_transformation.denormalize.Common'))
step_0.add_argument(name = 'inputs', argument_type = ArgumentType.CONTAINER, data_reference = 'inputs.0')






