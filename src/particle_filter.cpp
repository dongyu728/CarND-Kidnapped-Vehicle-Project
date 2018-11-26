#include <random>
#include <algorithm>
#include <random>
#include <iostream>
#include <tuple>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using std::normal_distribution;
using std::default_random_engine;

using vector_t = std::vector<double>;

default_random_engine gen;

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  double sample_x, sample_y, sample_psi;

  num_particles = 200;
  weights.resize(num_particles, 1.0f);

  for(unsigned i=0; i<num_particles; i++)
  {
    Particle p;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.id = i;
    p.weight = 1.0f;
    particles.push_back(p);
  }
  is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  normal_distribution<double> N_x(0, std_pos[0]);
  normal_distribution<double> N_y(0, std_pos[1]);
  normal_distribution<double> N_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    if (fabs(yaw_rate) < 0.00001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add noise
    particles[i].x += N_x(gen);
    particles[i].y += N_y(gen);
    particles[i].theta += N_theta(gen);
  }
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

  double min_dist, dist, dx, dy;
  int min_i;

  for(unsigned i = 0; i < observations.size(); i++)
  {
    auto obs = observations[i];

    min_dist = 9999999;
    min_i = -1;
    for(unsigned j = 0; j < predicted.size(); j++)
    {
      auto pred_xy= predicted[j];
      dx = (pred_xy.x - obs.x);
      dy = (pred_xy.y - obs.y);
      dist = dx*dx + dy*dy;
      if(dist < min_dist)
      {
        min_dist = dist;
        min_i = j;
      }
    }
    observations[i].id = min_i;
  }
}

const LandmarkObs convert_to_map(const LandmarkObs& obs, const Particle& p)
{
  LandmarkObs result;

  // First rotate the local coordinates to the right orientation
  result.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
  result.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
  result.id = obs.id;
  return result;
}

inline const double gaussian_2d(const LandmarkObs& obs, const LandmarkObs &lm, const double sigma[])
{
  auto cov_x = sigma[0]*sigma[0];
  auto cov_y = sigma[1]*sigma[1];
  auto normalizer = 2.0*M_PI*sigma[0]*sigma[1];
  auto dx = (obs.x - lm.x);
  auto dy = (obs.y - lm.y);
  return exp(-(dx*dx/(2*cov_x) + dy*dy/(2*cov_y)))/normalizer;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],std::vector<LandmarkObs> observations, Map map_landmarks) {
  
  double sigma_landmark [2] = {0.3, 0.3}; 
  double dist_range = sensor_range*sensor_range;

  for(unsigned p_ctr=0; p_ctr < particles.size(); p_ctr++)
  {
    auto p = particles[p_ctr];

    std::vector<LandmarkObs> predicted_landmarks;

    for(auto map_lm : map_landmarks.landmark_list)
    {
      LandmarkObs map_pred;
	  map_pred.x = map_lm.x_f;
	  map_pred.y = map_lm.y_f;
	  map_pred.id = map_lm.id_i;
	  auto dx = map_pred.x - p.x;
	  auto dy = map_pred.y - p.y;

      if(dx*dx + dy*dy <=dist_range )
		  predicted_landmarks.push_back(map_pred);
    }
    std::vector<LandmarkObs> map_obs;
    double total_prob = 1.0;

    for(auto obs_lm : observations)
    {
		auto obs_map = convert_to_map(obs_lm, p);
		map_obs.push_back(std::move(obs_map));
    }

	dataAssociation(predicted_landmarks, map_obs);

	for (unsigned i = 0; i < map_obs.size(); i++)
    {
		auto obs = map_obs[i];
      // Assume sorted by id and starting at 1
      auto assoc_lm = predicted_landmarks[obs.id];

      double prob_weight = gaussian_2d(obs, assoc_lm, sigma_landmark);
	  total_prob *= prob_weight;
    }
    particles[p_ctr].weight = total_prob;
    weights[p_ctr] = total_prob;
  }
  std::cout<<std::endl;
}

void ParticleFilter::resample() {

	  vector<double> weights;
	  double maxWeight = -9999999;
	  for (int i = 0; i < num_particles; i++) {
		  weights.push_back(particles[i].weight);
		  if (particles[i].weight > maxWeight) {
			  maxWeight = particles[i].weight;
		  }
	  }
	  std::discrete_distribution<double> distDouble(0.0, maxWeight);
	  std::discrete_distribution<int> distInt(0, num_particles - 1);

	  int index = distInt(gen);
	  double beta = 0.0;

	  vector<Particle> new_Particles;
	  for (int i = 0; i < num_particles; i++) {
		  beta += distDouble(gen) * 2.0;
		  while (beta > weights[index]) {
			  beta -= weights[index];
			  index = (index + 1) % num_particles;
		  }
		  new_Particles.push_back(particles[index]);
	  }

	  particles = new_Particles;
  }


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
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
