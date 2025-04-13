#ifndef SR_H
#define SR_H

#define A 0
#define B 1

struct msg {
    char data[20];
};

struct pkt {
    int seqnum;
    int acknum;
    int checksum;
    char payload[20];
};

// Sender A interface
void A_output(struct msg message);
void A_input(struct pkt packet);
void A_timerinterrupt(void);
void A_init(void);

// Receiver B interface
void B_input(struct pkt packet);
void B_init(void);

// Simulation API (provided by emulator)
void tolayer3(int AorB, struct pkt packet);
void tolayer5(int AorB, char data[20]);
void starttimer(int AorB, float increment);
void stoptimer(int AorB);

#endif
