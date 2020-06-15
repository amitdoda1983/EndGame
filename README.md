# EndGame
EVA3 Final Assignment


below are the rewards configured

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
                
