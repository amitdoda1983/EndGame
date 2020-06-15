# EndGame
EVA3 Final Assignment

This is the state being captured and used 

state = [self.car.view, orientation, -orientation, last_distance - self.distance]

car view is 28x28 image crop from car location with car as center.Orientation is angle with the target, last distance - self.distance is the change in the distance from target as compared to previous timestep.

below is how the rewards are configured

        # moving on the sand
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            last_reward = -5


        else:  # moving on the road
            self.car.velocity = Vector(1.5, 0).rotate(self.car.angle)
            last_reward = -2

            # moving towards the goal
            if self.distance < last_distance:
                last_reward = 1
                

The challenge i see is that the car is not taking the shorter route always, looks like further finetuning with rewards is needed, makng use of living penalty properly.
