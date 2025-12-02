import { Target, Users, Award } from "lucide-react";

const About = () => {
  const stats = [
    {
      icon: Target,
      value: "99%",
      label: "Accuracy Rate",
      description: "Industry-leading chord detection precision"
    },
    {
      icon: Users,
      value: "10K+",
      label: "Active Users",
      description: "Musicians worldwide trust our platform"
    },
    {
      icon: Award,
      value: "50M+",
      label: "Songs Analyzed",
      description: "Tracks processed with our AI technology"
    }
  ];

  return (
    <section id="about" className="py-24 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            {/* Content */}
            <div>
              <h2 className="font-sora font-bold text-4xl md:text-5xl mb-6">
                Revolutionizing Music
                <span className="text-primary"> Learning</span>
              </h2>
              <p className="text-muted-foreground text-lg mb-8 leading-relaxed">
                ChordAI was built by musicians, for musicians. We understand the frustration of 
                trying to figure out complex chord progressions by ear. That's why we created 
                the most advanced AI-powered chord detection platform available today.
              </p>
              <p className="text-muted-foreground text-lg mb-8 leading-relaxed">
                Whether you're a beginner learning your first songs or a professional producer 
                working on your next hit, ChordAI gives you the tools to understand any piece 
                of music instantly.
              </p>
              
              {/* Features List */}
              <div className="space-y-4">
                {[
                  "Advanced AI trained on millions of songs",
                  "Real-time chord detection and visualization", 
                  "Support for complex jazz and extended chords",
                  "Export options for sheet music and MIDI"
                ].map((feature, index) => (
                  <div key={index} className="flex items-center space-x-3">
                    <div className="w-2 h-2 rounded-full bg-primary" />
                    <span className="text-foreground">{feature}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Stats */}
            <div className="space-y-8">
              {stats.map((stat, index) => (
                <div key={index} className="glass-card p-6 group hover:scale-105 transition-bounce">
                  <div className="flex items-center space-x-4">
                    <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-smooth">
                      <stat.icon className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <div className="font-sora font-bold text-3xl text-primary">
                        {stat.value}
                      </div>
                      <div className="font-semibold text-foreground">
                        {stat.label}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {stat.description}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;