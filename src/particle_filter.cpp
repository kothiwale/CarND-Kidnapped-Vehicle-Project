#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  num_particles = 10;
  default_random_engine gen;
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  //initialize x,y, theta and weights
  for (int i = 0; i < num_particles; i++) {
    Particle P;
    P.x = dist_x(gen);
    P.y = dist_y(gen);
    P.theta = dist_theta(gen);
    P.weight = 1.0;
    // appending to particles
    particles.push_back(P);
    // Correspondiing weights
		weights.push_back(P.weight);
	}
  is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;
  for (int i = 0; i < num_particles; i++) {
    //x, y and the yaw angle when the yaw rate is not equal to zero
     
    // check out staraight driving
    if (fabs(yaw_rate) < 0.001) {
        yaw_rate = 0.001;
    }

    normal_distribution<double> dist_x(0.0, std_pos[0]);
		normal_distribution<double> dist_y(0.0, std_pos[1]);
		normal_distribution<double> dist_theta(0.0, std_pos[2]);
    
    double factor = (velocity/yaw_rate);
    particles[i].x += dist_x(gen) + factor * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
    particles[i].y += dist_y(gen) + factor * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
    particles[i].theta += dist_theta(gen) + delta_t * yaw_rate;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations){
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {
    double threshold = 1000000.0;
    int near_id = 0;
    for (int j=0; j< predicted.size();j++){
      
      //distance between predicted measurements and observations 
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (distance < threshold){
        threshold = distance;
        near_id = predicted[j].id;
      }
    }
    observations[i].id = near_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  for (int i=0; i< num_particles; i++){
  
    double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
    
    vector<LandmarkObs> predictions;
    
    //landmarks in sensor range
    for (int j=0; j<map_landmarks.landmark_list.size(); j++){
      if (fabs(map_landmarks.landmark_list[j].x_f - p_x) <= sensor_range && fabs(map_landmarks.landmark_list[j].y_f - p_y) <= sensor_range){
        LandmarkObs predict;
        predict.id = map_landmarks.landmark_list[j].id_i;
        predict.x = map_landmarks.landmark_list[j].x_f;
        predict.y = map_landmarks.landmark_list[j].y_f;
        predictions.push_back(predict);
			}
    }
  
    
    vector<LandmarkObs> observation_transformed;
    
    //transformation
    for (int j=0; j<observations.size(); j++){
			LandmarkObs o_transformed;
      o_transformed.id = observations[j].id;
			o_transformed.x = p_x + cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y;
			o_transformed.y = p_y + sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y;
			observation_transformed.push_back(o_transformed);
		}
    
    
    dataAssociation(predictions, observation_transformed);
    
		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];
		particles[i].weight = 1.0;
    
    
    //updating weights
		for (int j = 0; j < observation_transformed.size(); j++) {
			double mu_x, mu_y;
			for (int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == observation_transformed[j].id) {
					mu_x = predictions[k].x;
					mu_y = predictions[k].y;
				}
			}
      
      double x_obs_diff = observation_transformed[j].x - mu_x;
      double y_obs_diff = observation_transformed[j].y - mu_y;
      
      double gauss_norm = 1/(2 * M_PI * sig_x * sig_y);
      double exponent = ((x_obs_diff)*(x_obs_diff))/(2 * sig_x * sig_x) + ((y_obs_diff)*(y_obs_diff))/(2 * sig_y * sig_y);
			particles[i].weight *= (gauss_norm * exp(-exponent));
      weights[i] = particles[i].weight;
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine gen;
  discrete_distribution<int> dist(weights.begin(), weights.end());
  vector<Particle> new_particles;

  for (int i = 0; i < num_particles; i++) {
		new_particles.push_back(particles[dist(gen)]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
