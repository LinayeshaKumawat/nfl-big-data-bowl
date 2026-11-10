# The Route Lie Detector

The Route Lie Detector is an NFL tracking-data visualization tool that identifies moments when wide receivers make sharp directional cuts and gain separation from defenders in man coverage, helping broadcasters and coaches discover interesting plays.

Using Python and player-tracking data from seven NFL games, we analyze receiver movements, estimate defender reaction times, and measure changes in separation following a cut.

Our analysis identified 21 valid cuts across 353 receiver routes, including six high- or medium-confidence candidates with positive separation gains.

One notable example features Nico Collins gaining 0.67 yards of separation against C.J. Henderson following a 99° directional cut, with a measured defender reaction delay of 0.2 seconds.

Because defender assignments are estimated using movement patterns rather than explicitly labeled, our tool identifies potential deception moments rather than proving that a defender was fooled.
